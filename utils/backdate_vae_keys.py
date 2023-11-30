import os
import torch
from safetensors.torch import save_file, load_file

def fix_vae_keys(state_dict):
    new_state_dict = {}

    for key in state_dict.keys():
        new_key = key
        if key.startswith("first_stage_model"):
            if ".to_q" in key:
                print(f" *  backdating {key}")
                new_key = new_key.replace('.to_q.', '.q.')
                print(f" ** new key -> {new_key}\n")
            elif ".to_k" in key:
                print(f" *  backdating {key}")
                new_key = new_key.replace('.to_k.', '.k.')
                print(f" ** new key -> {new_key}\n")
            elif ".to_v" in key:
                print(f" *  backdating {key}")
                new_key = new_key.replace('.to_v.', '.v.')
                print(f" ** new key -> {new_key}\n")
            elif ".to_out.0" in key:
                print(f" *  backdating {key}")
                new_key = new_key.replace('.to_out.0', '.proj_out')
                print(f" ** new key -> {new_key}\n")

        new_state_dict[new_key] = state_dict[key]

    return new_state_dict


def _backdate_keys(filepath, state_dict):
    new_state_dict = fix_vae_keys(state_dict)
    base_path_without_ext = os.path.splitext(filepath)[0]
    ext = os.path.splitext(filepath)[1]

    new_path = f"{base_path_without_ext}_fixed{ext}"
    print (f"Saving to {new_path}")
    save_file(new_state_dict, new_path)

def _compare_keys(filea_state_dict, fileb_state_dict):
    #remove cond_stage_model.transformer.text_model.embeddings.position_ids key
    if 'cond_stage_model.transformer.text_model.embeddings.position_ids' in filea_state_dict:
        filea_state_dict.pop('cond_stage_model.transformer.text_model.embeddings.position_ids', None)
    if 'cond_stage_model.transformer.text_model.embeddings.position_ids' in fileb_state_dict:
        fileb_state_dict.pop('cond_stage_model.transformer.text_model.embeddings.position_ids', None)

    # sort the keys for comparison (best shot we have at a fair comparison without trying to count params and other nonsense)
    filea_state_dict_keys = sorted(filea_state_dict.keys())
    fileb_state_dict_keys = sorted(fileb_state_dict.keys())

    print("filea keys         <----->          fileb keys")
    # compare the keys line by line
    for filea_key, fileb_key in zip(filea_state_dict_keys, fileb_state_dict_keys):
        if filea_key != fileb_key:
            print("Mismatched keys:")
            print (f"   filea key: {filea_key}")
            print (f"   fileb key: {fileb_key}")            
        else:
            #print (f"{ckpt_key} == {st_key}")
            pass
    print("filea keys         <----->          fileb keys")

def _load(filepath):
    if filepath.endswith(".safetensors"):
        print(f" Loading {filepath} loading as safetensors file")
        state_dict = load_file(filepath)
    else: # LDM ckpt
        print(f" Loading {filepath} loading as LDM checkpoint")
        state_dict = torch.load(filepath, map_location='cpu')['state_dict']
    return state_dict

if __name__ == "__main__":
    print("BACKDATE AutoencoderKL/VAE KEYS TO OLD NAMES SCRIPT OF DOOM")
    print("================================")
    print(" --filea <path to ckpt or safetensors file>    file to backdate the VAE keys (will make a copy as <filename>_fixed.safetensors)")
    print(" --fileb <path to ckpt or safetensors file>    to compare keys to filea")
    print(" --compare                                     to run keys comparison (requires both --filea and --fileb)")
    print(" --backdate                                    to backdate keys (only for --filea)")
    print(" You must specify either --compare or --backdate to do anything")
    print(" ex.   python utils/backdate_vae_keys.py --filea my_finetune.safetensors --fileb original_sd15.ckpt --compare")
    print(" ex.   python utils/backdate_vae_keys.py --filea my_finetune.safetensors --backdate")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filea", type=str, required=True, help="Path to the safetensors file to fix")
    parser.add_argument("--fileb", type=str, required=False, help="Path to the safetensors file to fix")
    parser.add_argument("--compare", action="store_true", help="Compare keys")
    parser.add_argument("--backdate", action="store_true", help="backdates the keys in filea only")
    args = parser.parse_args()

    filea_state_dict = _load(args.filea) if args.filea else None
    fileb_state_dict = _load(args.fileb) if args.fileb else None

    if args.compare and not args.backdate:
        print(f"Comparing keys in {args.filea} to {args.fileb}")
        _compare_keys(filea_state_dict, fileb_state_dict)
    elif args.backdate:
        print(f"Backdating keys in {args.filea}")
        print(f"   ** ignoring {args.fileb}") if args.fileb else None
        _backdate_keys(args.filea, filea_state_dict)
    else:
        print("Please specify only --compare with both --filea and --fileb to compare keys and print differences to console")
        print(" ... or --backdate with only --filea to backdate its keys to old LDM names and save to <filea>_fixed.safetensors")