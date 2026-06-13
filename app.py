import argparse
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union, Optional

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from huggingface_hub import hf_hub_download
from transformers import DynamicCache
from safetensors.torch import load_file as safe_load_file

import MIDI
from midi_model import MIDIModel, config_name_list, MIDIModelConfig
from midi_synthesizer import MidiSynthesizer
from midi_tokenizer import MIDITokenizerV1, MIDITokenizerV2

MAX_SEED = np.iinfo(np.int32).max


@torch.inference_mode()
def generate(prompt=None, batch_size=1, max_len=512, temp=1.0, top_p=0.98, top_k=20,
             disable_patch_change=False, disable_control_change=False,
             disable_channels=None, disable_tracks=None,
             original_events=None, generator=None):
    if disable_channels is not None:
        disable_channels = [tokenizer.parameter_ids["channel"][c] for c in disable_channels]
    else:
        disable_channels = []
    if disable_tracks is not None:
        disable_tracks = [tokenizer.parameter_ids["track"][t] for t in disable_tracks]
    else:
        disable_tracks = []

    max_token_seq = tokenizer.max_token_seq
    if prompt is None:
        input_tensor = torch.full((1, max_token_seq), tokenizer.pad_id, dtype=torch.long, device=model.device)
        input_tensor[0, 0] = tokenizer.bos_id  # bos
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = torch.cat([input_tensor] * batch_size, dim=0)
    else:
        if len(prompt.shape) == 2:
            prompt = prompt[None, :]
            prompt = np.repeat(prompt, repeats=batch_size, axis=0)
        elif prompt.shape[0] == 1:
            prompt = np.repeat(prompt, repeats=batch_size, axis=0)
        elif len(prompt.shape) != 3 or prompt.shape[0] != batch_size:
            raise ValueError(f"invalid shape for prompt, {prompt.shape}")
        prompt = prompt[..., :max_token_seq]
        if prompt.shape[-1] < max_token_seq:
            prompt = np.pad(prompt, ((0, 0), (0, 0), (0, max_token_seq - prompt.shape[-1])),
                            mode="constant", constant_values=tokenizer.pad_id)
        input_tensor = torch.from_numpy(prompt).to(dtype=torch.long, device=model.device)
    input_tensor = input_tensor[:, -4096:]

    last_abs_time = [0] * batch_size
    for b in range(batch_size):
        t1_acc = 0
        for tokens in input_tensor[b]:
            event = tokenizer.tokens2event(tokens.tolist())
            if event:
                t1_acc += event[1]
                last_abs_time[b] = t1_acc * 16 + event[2]

    cur_len = input_tensor.shape[1]
    bar = tqdm.tqdm(desc="generating", total=max_len - cur_len)
    cache1 = DynamicCache()
    past_len = 0
    with bar:
        while cur_len < max_len:
            end = [False] * batch_size
            hidden = model.forward(input_tensor[:, past_len:], cache=cache1)[:, -1]
            proposed_tokens = None
            proposed_event_names = [""] * batch_size
            proposed_end = [False] * batch_size
            cache2 = DynamicCache()
            for i in range(max_token_seq):
                mask = torch.zeros((batch_size, tokenizer.vocab_size), dtype=torch.long, device=model.device)
                for b in range(batch_size):
                    if proposed_end[b]:
                        mask[b, tokenizer.pad_id] = 1
                        continue
                    if i == 0:
                        mask_ids = list(tokenizer.event_ids.values()) + [tokenizer.eos_id]
                        if disable_patch_change:
                            if tokenizer.event_ids["patch_change"] in mask_ids:
                                mask_ids.remove(tokenizer.event_ids["patch_change"])
                        if disable_control_change:
                            if tokenizer.event_ids["control_change"] in mask_ids:
                                mask_ids.remove(tokenizer.event_ids["control_change"])
                        mask[b, mask_ids] = 1
                    else:
                        param_names = tokenizer.events[proposed_event_names[b]]
                        if i > len(param_names):
                            mask[b, tokenizer.pad_id] = 1
                            continue
                        p_name = param_names[i - 1]
                        mask_ids = tokenizer.parameter_ids[p_name]
                        if p_name == "channel" and disable_channels is not None:
                            mask_ids = [idx for idx in mask_ids if idx not in disable_channels]
                        if p_name == "track" and disable_tracks is not None:
                            mask_ids = [idx for idx in mask_ids if idx not in disable_tracks]
                        mask[b, mask_ids] = 1
                mask = mask.unsqueeze(1)
                x = proposed_tokens
                if i != 0:
                    temp_hidden = None
                    x = x[:, -1:]
                else:
                    temp_hidden = hidden
                logits = model.forward_token(temp_hidden, x, cache=cache2)[:, -1:]
                scores = torch.softmax(logits / temp, dim=-1) * mask
                samples = model.sample_top_p_k(scores, top_p, top_k, generator=generator)
                if i == 0:
                    proposed_tokens = samples
                    for b in range(batch_size):
                        if proposed_end[b]: continue
                        eid = samples[b].item()
                        if eid == tokenizer.eos_id: proposed_end[b] = True
                        else: proposed_event_names[b] = tokenizer.id_events[eid]
                else:
                    proposed_tokens = torch.cat([proposed_tokens, samples], dim=1)
                    if all([len(tokenizer.events[proposed_event_names[b]]) == i for b in range(batch_size) if not proposed_end[b]]):
                        break

            if proposed_tokens.shape[1] < max_token_seq:
                proposed_tokens = F.pad(proposed_tokens, (0, max_token_seq - proposed_tokens.shape[1]),
                                        "constant", value=tokenizer.pad_id)

            # Interleave logic
            next_token_seq = proposed_tokens.clone()
            end = [False] * batch_size
            for b in range(batch_size):
                e_prop = tokenizer.tokens2event(proposed_tokens[b].tolist())
                t_prev = last_abs_time[b]
                t1_prev = t_prev // 16
                
                if e_prop:
                    t_prop = (t1_prev + e_prop[1]) * 16 + e_prop[2]
                    t_prop = max(t_prop, t_prev)  # Safety clamp
                else:
                    t_prop = float('inf')
                
                if original_events is not None and original_events[b]:
                    e_orig_data = original_events[b][0]
                    t_orig = e_orig_data["abs_time"]
                    
                    if t_orig <= t_prop:
                        # Use original event instead of proposed
                        ev = list(e_orig_data["event"])
                        # Correct delta_t1 and abs_t2 for the tokenizer format
                        ev[1] = (t_orig // 16) - t1_prev
                        ev[2] = t_orig % 16
                        
                        tokens = tokenizer.event2tokens(ev)
                        if len(tokens) < max_token_seq:
                            tokens += [tokenizer.pad_id] * (max_token_seq - len(tokens))
                        next_token_seq[b] = torch.tensor(tokens, device=model.device, dtype=torch.long)
                        
                        original_events[b].pop(0)
                        last_abs_time[b] = t_orig
                    else:
                        # Use proposed event
                        last_abs_time[b] = t_prop
                        if proposed_end[b]: end[b] = True
                else:
                    # No original events left, use proposed
                    if e_prop:
                        last_abs_time[b] = t_prop
                    if proposed_end[b]: end[b] = True

            next_token_seq = next_token_seq.unsqueeze(1)
            input_tensor = torch.cat([input_tensor, next_token_seq], dim=1)
            past_len = cur_len
            cur_len += 1
            bar.update(1)
            yield next_token_seq[:, 0].cpu().numpy()
            if all(end):
                break


def create_msg(name, data):
    return {"name": name, "data": data}


def send_msgs(msgs):
    return json.dumps(msgs)


def run(tab, mid_seq, continuation_state, continuation_select, instruments, drum_kit, bpm, time_sig, key_sig, mid,
        midi_events, add_track_mid, add_track_midi_events, add_track_reduce_cc_st, add_track_remap_track_channel,
        add_track_add_default_instr, add_track_remove_empty_channels,
        add_track_context_events,
        add_track_instruments, add_track_drum_kit,
        reduce_cc_st, remap_track_channel, add_default_instr, remove_empty_channels,
        seed, seed_rand, gen_events, temp, top_p, top_k, allow_cc):
    bpm = int(bpm)
    original_events = None
    disable_tracks = None
    if time_sig == "auto":
        time_sig = None
        time_sig_nn = 4
        time_sig_dd = 2
    else:
        time_sig_nn, time_sig_dd = time_sig.split('/')
        time_sig_nn = int(time_sig_nn)
        time_sig_dd = {2: 1, 4: 2, 8: 3}[int(time_sig_dd)]
    if key_sig == 0:
        key_sig = None
        key_sig_sf = 0
        key_sig_mi = 0
    else:
        key_sig = (key_sig - 1)
        key_sig_sf = key_sig // 2 - 7
        key_sig_mi = key_sig % 2
    gen_events = int(gen_events)
    max_len = gen_events
    if seed_rand:
        seed = np.random.randint(0, MAX_SEED)
    generator = torch.Generator(opt.device).manual_seed(seed)
    disable_patch_change = False
    disable_channels = None
    if tab == 0:
        i = 0
        mid = [[tokenizer.bos_id] + [tokenizer.pad_id] * (tokenizer.max_token_seq - 1)]
        if tokenizer.version == "v2":
            if time_sig is not None:
                mid.append(tokenizer.event2tokens(["time_signature", 0, 0, 0, time_sig_nn - 1, time_sig_dd - 1]))
            if key_sig is not None:
                mid.append(tokenizer.event2tokens(["key_signature", 0, 0, 0, key_sig_sf + 7, key_sig_mi]))
        if bpm != 0:
            mid.append(tokenizer.event2tokens(["set_tempo", 0, 0, 0, bpm]))
        patches = {}
        if instruments is None:
            instruments = []
        for instr in instruments:
            patches[i] = patch2number[instr]
            i = (i + 1) if i != 8 else 10
        if drum_kit != "None":
            patches[9] = drum_kits2number[drum_kit]
        for i, (c, p) in enumerate(patches.items()):
            mid.append(tokenizer.event2tokens(["patch_change", 0, 0, i + 1, c, p]))
        mid = np.asarray([mid] * OUTPUT_BATCH_SIZE, dtype=np.int64)
        mid_seq = mid.tolist()
        if len(instruments) > 0:
            disable_patch_change = True
            disable_channels = [i for i in range(16) if i not in patches]
    elif tab == 1 and mid is not None:
        eps = 4 if reduce_cc_st else 0
        mid = tokenizer.tokenize(MIDI.midi2score(mid), cc_eps=eps, tempo_eps=eps,
                                 remap_track_channel=remap_track_channel,
                                 add_default_instr=add_default_instr,
                                 remove_empty_channels=remove_empty_channels)
        midi_events = int(midi_events)
        if midi_events <= 4096:
            mid = mid[:midi_events]
        mid = np.asarray([mid] * OUTPUT_BATCH_SIZE, dtype=np.int64)
        mid_seq = mid.tolist()
    elif tab == 2 and mid_seq is not None:
        mid = np.asarray(mid_seq, dtype=np.int64)
        if continuation_select > 0:
            continuation_state.append(mid_seq)
            mid = np.repeat(mid[continuation_select - 1:continuation_select], repeats=OUTPUT_BATCH_SIZE, axis=0)
            mid_seq = mid.tolist()
        else:
            continuation_state.append(mid.shape[1])
    elif tab == 3 and add_track_mid is not None:
        eps = 4 if add_track_reduce_cc_st else 0
        all_tokens = tokenizer.tokenize(MIDI.midi2score(add_track_mid), cc_eps=eps, tempo_eps=eps,
                                        remap_track_channel=add_track_remap_track_channel,
                                        add_default_instr=add_track_add_default_instr,
                                        remove_empty_channels=add_track_remove_empty_channels)
        limit = int(add_track_midi_events)
        if limit <= 4096: all_tokens = all_tokens[:limit]

        prompt = []
        all_music_events = []
        t1_acc = 0
        for tokens in all_tokens:
            event = tokenizer.tokens2event(tokens)
            if not event: continue
            t1_acc += event[1]
            abs_t = t1_acc * 16 + event[2]
            
            # Separate setup tokens (BOS or meta at t=0) from actual music events
            if event[0] == "bos" or (abs_t == 0 and event[0] not in ["note", "eos"]):
                prompt.append(tokens)
            elif event[0] != "eos":
                all_music_events.append({"tokens": tokens, "event": event, "abs_time": abs_t})

        # Take n events as context (added to prompt, Proposal phase disabled for these)
        n_context = int(add_track_context_events)
        for i in range(min(n_context, len(all_music_events))):
            prompt.append(all_music_events[i]["tokens"])
        
        music_events = [{"event": e["event"], "abs_time": e["abs_time"]} for e in all_music_events[n_context:]]

        used_channels = set()
        used_tracks = set()
        for tokens in all_tokens:
            ev = tokenizer.tokens2event(tokens)
            if not ev: continue
            if ev[0] in ["note", "patch_change", "control_change"]:
                c_idx = 5 if (ev[0] == "note" and tokenizer.version == "v1") else 4
                if c_idx < len(ev): used_channels.add(ev[c_idx])
            used_tracks.add(ev[3])

        new_channels = []
        new_tracks = []
        max_track = max(list(used_tracks) + [0])

        if add_track_instruments:
            next_channel = 0
            for instr in add_track_instruments:
                while next_channel in used_channels or next_channel == 9: next_channel += 1
                if next_channel > 15: break
                max_track += 1
                prompt.append(tokenizer.event2tokens(["patch_change", 0, 0, max_track, next_channel, patch2number[instr]]))
                used_channels.add(next_channel)
                new_channels.append(next_channel)
                new_tracks.append(max_track)
        if add_track_drum_kit != "None" and 9 not in used_channels:
            max_track += 1
            prompt.append(tokenizer.event2tokens(["patch_change", 0, 0, max_track, 9, drum_kits2number[add_track_drum_kit]]))
            new_channels.append(9)
            new_tracks.append(max_track)

        if len(new_channels) > 0:
            disable_patch_change = True
            disable_channels = [c for c in range(16) if c not in new_channels]
            disable_tracks = [t for t in range(128) if t not in new_tracks]

        mid = np.asarray([prompt] * OUTPUT_BATCH_SIZE, dtype=np.int64)
        mid_seq = mid.tolist()
        original_events = [music_events[:] for _ in range(OUTPUT_BATCH_SIZE)]
    else:
        continuation_state = [0]
        mid = [[tokenizer.bos_id] + [tokenizer.pad_id] * (tokenizer.max_token_seq - 1)]
        mid = np.asarray([mid] * OUTPUT_BATCH_SIZE, dtype=np.int64)
        mid_seq = mid.tolist()

    if mid is not None:
        max_len += mid.shape[1]

    init_msgs = [create_msg("progress", [0, gen_events])]
    if not (tab == 2 and continuation_select == 0):
        for i in range(OUTPUT_BATCH_SIZE):
            events = [tokenizer.tokens2event(tokens) for tokens in mid_seq[i]]
            init_msgs += [create_msg("visualizer_clear", [i, tokenizer.version]),
                          create_msg("visualizer_append", [i, events])]
    yield mid_seq, continuation_state, seed, send_msgs(init_msgs)
    midi_generator = generate(mid, batch_size=OUTPUT_BATCH_SIZE, max_len=max_len, temp=temp,
                              top_p=top_p, top_k=top_k, disable_patch_change=disable_patch_change,
                              disable_control_change=not allow_cc, disable_channels=disable_channels,
                              disable_tracks=disable_tracks,
                              original_events=original_events,
                              generator=generator)
    events = [list() for i in range(OUTPUT_BATCH_SIZE)]
    t = time.time()
    for i, token_seqs in enumerate(midi_generator):
        token_seqs = token_seqs.tolist()
        for j in range(OUTPUT_BATCH_SIZE):
            token_seq = token_seqs[j]
            mid_seq[j].append(token_seq)
            events[j].append(tokenizer.tokens2event(token_seq))
        if time.time() - t > 0.2:
            msgs = [create_msg("progress", [i + 1, gen_events])]
            for j in range(OUTPUT_BATCH_SIZE):
                msgs += [create_msg("visualizer_append", [j, events[j]])]
                events[j] = list()
            yield mid_seq, continuation_state, seed, send_msgs(msgs)
            t = time.time()
    yield mid_seq, continuation_state, seed, send_msgs([])


def finish_run(mid_seq):
    if mid_seq is None:
        outputs = [None] * OUTPUT_BATCH_SIZE
        return *outputs, []
    outputs = []
    end_msgs = [create_msg("progress", [0, 0])]
    if not os.path.exists("outputs"):
        os.mkdir("outputs")
    for i in range(OUTPUT_BATCH_SIZE):
        events = [tokenizer.tokens2event(tokens) for tokens in mid_seq[i]]
        mid = tokenizer.detokenize(mid_seq[i])
        with open(f"outputs/output{i + 1}.mid", 'wb') as f:
            f.write(MIDI.score2midi(mid))
        outputs.append(f"outputs/output{i + 1}.mid")
        end_msgs += [create_msg("visualizer_clear", [i, tokenizer.version]),
                     create_msg("visualizer_append", [i, events]),
                     create_msg("visualizer_end", i)]
    return *outputs, send_msgs(end_msgs)


def synthesis_task(mid):
    return synthesizer.synthesis(MIDI.score2opus(mid))

def render_audio(mid_seq, should_render_audio):
    if (not should_render_audio) or mid_seq is None:
        outputs = [None] * OUTPUT_BATCH_SIZE
        return tuple(outputs)
    outputs = []
    if not os.path.exists("outputs"):
        os.mkdir("outputs")
    audio_futures = []
    for i in range(OUTPUT_BATCH_SIZE):
        mid = tokenizer.detokenize(mid_seq[i])
        audio_future = thread_pool.submit(synthesis_task, mid)
        audio_futures.append(audio_future)
    for future in audio_futures:
        outputs.append((44100, future.result()))
    if OUTPUT_BATCH_SIZE == 1:
        return outputs[0]
    return tuple(outputs)


def undo_continuation(mid_seq, continuation_state):
    if mid_seq is None or len(continuation_state) < 2:
        return mid_seq, continuation_state, send_msgs([])
    if isinstance(continuation_state[-1], list):
        mid_seq = continuation_state[-1]
    else:
        mid_seq = [ms[:continuation_state[-1]] for ms in mid_seq]
    continuation_state = continuation_state[:-1]
    end_msgs = [create_msg("progress", [0, 0])]
    for i in range(OUTPUT_BATCH_SIZE):
        events = [tokenizer.tokens2event(tokens) for tokens in mid_seq[i]]
        end_msgs += [create_msg("visualizer_clear", [i, tokenizer.version]),
                     create_msg("visualizer_append", [i, events]),
                     create_msg("visualizer_end", i)]
    return mid_seq, continuation_state, send_msgs(end_msgs)


def load_model(path, model_config, lora_path):
    global model, tokenizer
    if model_config == "auto":
        config_path = Path(path).parent / "config.json"
        if config_path.exists():
            config=MIDIModelConfig.from_json_file(config_path)
        else:
            return "can not find config.json, please specify config"
    else:
        config = MIDIModelConfig.from_name(model_config)
    model = MIDIModel(config=config)
    tokenizer = model.tokenizer
    if path.endswith(".safetensors"):
        state_dict = safe_load_file(path)
    else:
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    if lora_path:
        model = model.load_merge_lora(lora_path)
    model.to(opt.device, dtype=torch.bfloat16 if opt.device == "cuda" else torch.float32).eval()
    return "success"


def get_model_path():
    ckpt_files = glob.glob("**/*.ckpt", recursive=True)
    bin_files = glob.glob("**/*.bin", recursive=True)
    safetensors_files = glob.glob("**/*.safetensors", recursive=True)
    model_paths = sorted(ckpt_files + bin_files + safetensors_files)
    model_paths = [model_path for model_path in model_paths if "adapter_model" not in model_path]  # lora
    return gr.Dropdown(choices=model_paths)

def get_lora_path():
    lora_paths = sorted(glob.glob("**/adapter_config.json", recursive=True))
    lora_paths = [lora_path.replace("adapter_config.json","") for lora_path in lora_paths]
    return gr.Dropdown(choices=lora_paths)


def load_javascript(dir="javascript"):
    scripts_list = glob.glob(f"{dir}/*.js")
    javascript = ""
    for path in scripts_list:
        with open(path, "r", encoding="utf8") as jsfile:
            js_content = jsfile.read()
            js_content = js_content.replace("const MIDI_OUTPUT_BATCH_SIZE=4;",
                                            f"const MIDI_OUTPUT_BATCH_SIZE={OUTPUT_BATCH_SIZE};")
            javascript += f"\n<!-- {path} --><script>{js_content}</script>"
    template_response_ori = gr.routes.templates.TemplateResponse

    def template_response(*args, **kwargs):
        res = template_response_ori(*args, **kwargs)
        res.body = res.body.replace(
            b'</head>', f'{javascript}</head>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response


number2drum_kits = {-1: "None", 0: "Standard", 8: "Room", 16: "Power", 24: "Electric", 25: "TR-808", 32: "Jazz",
                    40: "Blush", 48: "Orchestra"}
patch2number = {v: k for k, v in MIDI.Number2patch.items()}
drum_kits2number = {v: k for k, v in number2drum_kits.items()}
key_signatures = ['C♭', 'A♭m', 'G♭', 'E♭m', 'D♭', 'B♭m', 'A♭', 'Fm', 'E♭', 'Cm', 'B♭', 'Gm', 'F', 'Dm',
                  'C', 'Am', 'G', 'Em', 'D', 'Bm', 'A', 'F♯m', 'E', 'C♯m', 'B', 'G♯m', 'F♯', 'D♯m', 'C♯', 'A♯m']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860, help="gradio server port")
    parser.add_argument("--device", type=str, default="cuda", help="device to run model")
    parser.add_argument("--batch", type=int, default=4, help="batch size")
    parser.add_argument("--share", action="store_true", default=False, help="share gradio")
    opt = parser.parse_args()
    OUTPUT_BATCH_SIZE = opt.batch
    soundfont_path = hf_hub_download(repo_id="skytnt/midi-model", filename="soundfont.sf2")
    synthesizer = MidiSynthesizer(soundfont_path)
    thread_pool = ThreadPoolExecutor(max_workers=OUTPUT_BATCH_SIZE)
    tokenizer: Union[MIDITokenizerV1, MIDITokenizerV2, None] = None
    model: Optional[MIDIModel] = None

    load_javascript()
    app = gr.Blocks()
    with app:
        js_msg_history_state = gr.State(value=[])
        js_msg = gr.Textbox(elem_id="msg_receiver", visible=False)
        js_msg.change(None, [js_msg], [], js="""
                (msg_json) =>{
                    let msgs = JSON.parse(msg_json);
                    executeCallbacks(msgReceiveCallbacks, msgs);
                    return [];
                }
                """)
        with gr.Accordion(label="Model option", open=True):
            load_model_path_btn = gr.Button("Get Models")
            model_path_input = gr.Dropdown(label="model")
            model_config_input = gr.Dropdown(label="config",choices=["auto"] + config_name_list, value="auto")
            load_model_path_btn.click(get_model_path, [], model_path_input)
            load_lora_path_btn = gr.Button("Get Loras")
            lora_path_input = gr.Dropdown(label="lora")
            load_lora_path_btn.click(get_lora_path, [], lora_path_input)
            load_model_btn = gr.Button("Load")
            model_msg = gr.Textbox()
            load_model_btn.click(
                load_model, [model_path_input, model_config_input, lora_path_input], model_msg
            )
        tab_select = gr.State(value=0)
        with gr.Tabs():
            with gr.TabItem("custom prompt") as tab1:
                input_instruments = gr.Dropdown(label="🪗instruments (auto if empty)", choices=list(patch2number.keys()),
                                                multiselect=True, max_choices=15, type="value")
                input_drum_kit = gr.Dropdown(label="🥁drum kit", choices=list(drum_kits2number.keys()), type="value",
                                             value="None")
                input_bpm = gr.Slider(label="BPM (beats per minute, auto if 0)", minimum=0, maximum=255,
                                      step=1,
                                      value=0)
                input_time_sig = gr.Radio(label="time signature (only for tv2 models)",
                                          value="auto",
                                          choices=["auto", "4/4", "2/4", "3/4", "6/4", "7/4",
                                                   "2/2", "3/2", "4/2", "3/8", "5/8", "6/8", "7/8", "9/8", "12/8"]
                                          )
                input_key_sig = gr.Radio(label="key signature (only for tv2 models)",
                                         value="auto",
                                         choices=["auto"] + key_signatures,
                                         type="index"
                                         )
                example1 = gr.Examples([
                    [[], "None"],
                    [["Acoustic Grand"], "None"],
                    [["Flute", "Synth Voice", "Pad 4 (Choir)", "Overdriven Guitar", "Distortion Guitar",
                      "Electric Bass(finger)"], "Standard"],
                    [['Acoustic Grand', 'SynthStrings 2', 'SynthStrings 1', 'Pizzicato Strings',
                      'Pad 2 (warm)', 'Tremolo Strings', 'String Ensemble 1'], "Orchestra"],
                    [['Trumpet', 'Oboe', 'Trombone', 'String Ensemble 1', 'Clarinet',
                      'French Horn', 'Pad 4 (choir)', 'Bassoon', 'Flute'], "None"],
                    [['Flute', 'French Horn', 'Clarinet', 'String Ensemble 2', 'English Horn', 'Bassoon',
                      'Oboe', 'Pizzicato Strings'], "Orchestra"],
                    [['Electric Piano 2', 'Lead 5 (charang)', 'Electric Bass(pick)', 'Lead 2 (sawtooth)',
                      'Pad 1 (new age)', 'Orchestra Hit', 'Cello', 'Electric Guitar(clean)'], "Standard"],
                    [["Electric Guitar(clean)", "Electric Guitar(muted)", "Overdriven Guitar", "Distortion Guitar",
                      "Electric Bass(finger)"], "Standard"]
                ], [input_instruments, input_drum_kit])
            with gr.TabItem("midi prompt") as tab2:
                input_midi = gr.File(label="input midi", file_types=[".midi", ".mid"], type="binary")
                input_midi_events = gr.Slider(label="use first n midi events as prompt (all if 4097)", minimum=1,
                                              maximum=4097,
                                              step=1,
                                              value=128)
                input_reduce_cc_st = gr.Checkbox(label="reduce control_change and set_tempo events", value=True)
                input_remap_track_channel = gr.Checkbox(
                    label="remap tracks and channels so each track has only one channel and in order", value=True)
                input_add_default_instr = gr.Checkbox(
                    label="add a default instrument to channels that don't have an instrument", value=True)
                input_remove_empty_channels = gr.Checkbox(label="remove channels without notes", value=False)
            with gr.TabItem("last output prompt") as tab3:
                gr.Markdown("Continue generating on the last output.")
                input_continuation_select = gr.Radio(label="select output to continue generating", value="all",
                                               choices=["all"] + [f"output{i + 1}" for i in range(OUTPUT_BATCH_SIZE)],
                                               type="index"
                                               )
                undo_btn = gr.Button("undo the last continuation")
            with gr.TabItem("add track") as tab4:
                gr.Markdown("Add a new track based on an uploaded MIDI file.")
                input_add_track_midi = gr.File(label="input midi", file_types=[".midi", ".mid"], type="binary")
                input_add_track_midi_events = gr.Slider(label="use first n midi events as prompt (all if 4097)",
                                                        minimum=1, maximum=4097, step=1, value=128)
                input_add_track_reduce_cc_st = gr.Checkbox(label="reduce control_change and set_tempo events",
                                                           value=True)
                input_add_track_remap_track_channel = gr.Checkbox(
                    label="remap tracks and channels so each track has only one channel and in order", value=True)
                input_add_track_add_default_instr = gr.Checkbox(
                    label="add a default instrument to channels that don't have an instrument", value=True)
                input_add_track_remove_empty_channels = gr.Checkbox(label="remove channels without notes",
                                                                    value=False)
                input_add_track_context_events = gr.Slider(label="context events from original (proposing disabled)",
                                                        minimum=0, maximum=4096, step=1, value=128)
                input_add_track_instruments = gr.Dropdown(label="🪗instruments to add", choices=list(patch2number.keys()),
                                                multiselect=True, max_choices=15, type="value")
                input_add_track_drum_kit = gr.Dropdown(label="🥁drum kit to add", choices=list(drum_kits2number.keys()), type="value",
                                             value="None")
                undo_btn_track = gr.Button("undo the last addition")

        tab1.select(lambda: 0, None, tab_select, queue=False)
        tab2.select(lambda: 1, None, tab_select, queue=False)
        tab3.select(lambda: 2, None, tab_select, queue=False)
        tab4.select(lambda: 3, None, tab_select, queue=False)
        input_seed = gr.Slider(label="seed", minimum=0, maximum=2 ** 31 - 1,
                               step=1, value=0)
        input_seed_rand = gr.Checkbox(label="random seed", value=True)
        input_gen_events = gr.Slider(label="generate max n midi events", minimum=1, maximum=4096,
                                     step=1, value=512)
        with gr.Accordion("options", open=False):
            input_temp = gr.Slider(label="temperature", minimum=0.1, maximum=1.2, step=0.01, value=1)
            input_top_p = gr.Slider(label="top p", minimum=0.1, maximum=1, step=0.01, value=0.94)
            input_top_k = gr.Slider(label="top k", minimum=1, maximum=128, step=1, value=20)
            input_allow_cc = gr.Checkbox(label="allow midi cc event", value=True)
            input_render_audio = gr.Checkbox(label="render audio after generation", value=True)
            example3 = gr.Examples([[1, 0.94, 128], [1, 0.98, 20], [1, 0.98, 12]],
                                   [input_temp, input_top_p, input_top_k])
        run_btn = gr.Button("generate", variant="primary")
        stop_btn = gr.Button("stop and output")
        output_midi_seq = gr.State()
        output_continuation_state = gr.State([0])
        midi_outputs = []
        audio_outputs = []
        with gr.Tabs(elem_id="output_tabs"):
            for i in range(OUTPUT_BATCH_SIZE):
                with gr.TabItem(f"output {i + 1}") as tab1:
                    output_midi_visualizer = gr.HTML(elem_id=f"midi_visualizer_container_{i}")
                    output_audio = gr.Audio(label="output audio", format="mp3", elem_id=f"midi_audio_{i}")
                    output_midi = gr.File(label="output midi", file_types=[".mid"])
                    midi_outputs.append(output_midi)
                    audio_outputs.append(output_audio)
        run_event = run_btn.click(run, [tab_select, output_midi_seq, output_continuation_state,
                                        input_continuation_select, input_instruments, input_drum_kit, input_bpm,
                                        input_time_sig, input_key_sig, input_midi, input_midi_events,
                                        input_add_track_midi,
                                        input_add_track_midi_events,
                                        input_add_track_reduce_cc_st,
                                        input_add_track_remap_track_channel,
                                        input_add_track_add_default_instr,
                                        input_add_track_remove_empty_channels,
                                        input_add_track_context_events,
                                        input_add_track_instruments,
                                        input_add_track_drum_kit,
                                        input_reduce_cc_st, input_remap_track_channel,
                                        input_add_default_instr, input_remove_empty_channels,
                                        input_seed, input_seed_rand, input_gen_events, input_temp, input_top_p,
                                        input_top_k, input_allow_cc],
                                  [output_midi_seq, output_continuation_state, input_seed, js_msg],
                                  concurrency_limit=10, queue=True)
        finish_run_event = run_event.then(fn=finish_run,
                                          inputs=[output_midi_seq],
                                          outputs=midi_outputs + [js_msg],
                                          queue=False)
        finish_run_event.then(fn=render_audio,
                              inputs=[output_midi_seq, input_render_audio],
                              outputs=audio_outputs,
                              queue=False)
        stop_btn.click(None, [], [], cancels=run_event, queue=False)
        undo_btn.click(undo_continuation, [output_midi_seq, output_continuation_state],
                       [output_midi_seq, output_continuation_state, js_msg], queue=False)
        undo_btn_track.click(undo_continuation, [output_midi_seq, output_continuation_state],
                       [output_midi_seq, output_continuation_state, js_msg], queue=False)
    # load_javascript not work on ssr mode
    app.launch(server_port=opt.port, inbrowser=True, share=opt.share, ssr_mode=False)
    thread_pool.shutdown()
