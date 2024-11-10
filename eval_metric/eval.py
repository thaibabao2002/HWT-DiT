from metrics import *
import os


def evaluate(parent_real_path, parent_fake_path, parent_real_ocr_path=None, parent_fake_ocr_path=None, device='cuda'):
    hwd_score, fid_score, cer_score = None, None, None
    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cpus = os.cpu_count()
    num_workers = min(num_cpus, 8) if num_cpus is not None else 0

    hwd_score = calculate_hwd_given_paths(parent_real_path, parent_fake_path, device=device)  ## eval with batch sampler
    # fid_score = calculate_fid_given_paths(paths=[parent_real_path, parent_fake_path], batch_size=50, 
    #                                       device=device, dims=2048, num_workers=num_workers, eval_type='padding') ## eval with batch padding images

    if parent_real_ocr_path is not None and parent_fake_ocr_path is not None:
        cer_score = calculate_cer_given_paths_txt(parent_real_ocr_path, parent_fake_ocr_path, cer_method="cer-torch")
    if cer_score:
        return {'hwd score': hwd_score,
                'fid score': fid_score,
                'cer score': cer_score}
    else:
        return {'hwd score': hwd_score,
                'fid score': fid_score}


parent_real_path = ""
parent_fake_path = ""

print(evaluate(parent_real_path, parent_fake_path, device='cuda'))
