import argparse
import logging
import sys
import os
import cv2
from tqdm import tqdm
import pytesseract
from torchmetrics.text import CharErrorRate
import editdistance

logging.basicConfig(
    format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


def levenshtein(u, v):
    prev = None
    curr = [0] + list(range(1, len(v) + 1))
    # Operations: (SUB, DEL, INS)
    prev_ops = None
    curr_ops = [(0, 0, i) for i in range(len(v) + 1)]
    for x in range(1, len(u) + 1):
        prev, curr = curr, [x] + ([None] * len(v))
        prev_ops, curr_ops = curr_ops, [(0, x, 0)] + ([None] * len(v))
        for y in range(1, len(v) + 1):
            delcost = prev[y] + 1
            addcost = curr[y - 1] + 1
            subcost = prev[y - 1] + int(u[x - 1] != v[y - 1])
            curr[y] = min(subcost, delcost, addcost)
            if curr[y] == subcost:
                (n_s, n_d, n_i) = prev_ops[y - 1]
                curr_ops[y] = (n_s + int(u[x - 1] != v[y - 1]), n_d, n_i)
            elif curr[y] == delcost:
                (n_s, n_d, n_i) = prev_ops[y]
                curr_ops[y] = (n_s, n_d + 1, n_i)
            else:
                (n_s, n_d, n_i) = curr_ops[y - 1]
                curr_ops[y] = (n_s, n_d, n_i + 1)
    return curr[len(v)], curr_ops[len(v)]


def load_file(fname, encoding):
    try:
        f = open(fname, 'r')
        data = []
        for line in f:
            data.append(line.rstrip('\n').rstrip('\r').decode(encoding))
        f.close()
    except:
        logging.error('Error reading file "%s"', fname)
        exit(1)
    return data


def cer_method_local(lst_text_real, lst_text_fake):
    cer_s, cer_i, cer_d, cer_n = 0, 0, 0, 0
    sen_err = 0
    for n in range(len(lst_text_real)):
        # update CER statistics
        _, (s, i, d) = levenshtein(lst_text_real[n], lst_text_fake[n])
        cer_s += s
        cer_i += i
        cer_d += d
        cer_n += len(lst_text_real[n])
    if cer_n > 0:
        return (cer_s + cer_i + cer_d) / cer_n
    else:
        return None


def cer_method_torch(lst_text_real, lst_text_fake):
    cer = CharErrorRate()
    cer_score = cer(lst_text_real, lst_text_fake)
    return cer_score


def cer_method_editdistance(lst_text_real, lst_text_fake):
    total_dist = 0
    total_len = 0
    for i in range(len(lst_text_real)):
        dis = float(editdistance.eval(lst_text_real[i], lst_text_fake[i]))
        total_dist += dis
        total_len += len(lst_text_real[i])

    if total_len > 0:
        return total_dist / total_len
    else:
        return None


def calculate_cer_given_paths(parent_real_path, parent_fake_path, ocr_model=None, cer_method='local', device='cpu'):
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    lst_fd = os.listdir(parent_real_path)
    lst_text_real = []
    lst_text_fake = []
    for fd_name in tqdm(lst_fd, desc="compute cer score:"):
        fd_path = os.path.join(parent_real_path, fd_name)
        lst_im = os.listdir(fd_path)
        for im_name in lst_im:
            real_im_path = os.path.join(fd_path, im_name)
            fake_im_path = os.path.join(parent_fake_path, fd_name, im_name)
            im_real = cv2.imread(real_im_path)
            im_real = cv2.cvtColor(im_real, cv2.COLOR_BGR2RGB)

            im_fake = cv2.imread(fake_im_path)
            im_fake = cv2.cvtColor(im_fake, cv2.COLOR_BGR2RGB)

            if ocr_model == None:
                ocr_real = pytesseract.image_to_string(im_real, config=custom_config)
                ocr_fake = pytesseract.image_to_string(im_fake, config=custom_config)
                lst_text_real.append(ocr_real)
                lst_text_fake.append(ocr_fake)
            else:
                ocr_model.to(device)
                ocr_real = ocr_model(im_real)
                ocr_fake = ocr_model(im_fake)
                lst_text_real.append(ocr_real)
                lst_text_fake.append(ocr_fake)

    if cer_method == 'local':
        return cer_method_local(lst_text_real, lst_text_fake)
    elif cer_method == "cer-torch":
        return cer_method_torch(lst_text_real, lst_text_fake)
    elif cer_method == 'editdistance':
        return cer_method_editdistance(lst_text_real, lst_text_fake)


def calculate_cer_given_paths_txt(parent_real_path, parent_fake_path, cer_method='local'):
    lst_fd = os.listdir(parent_real_path)
    lst_text_real = []
    lst_text_fake = []
    for fd_name in tqdm(lst_fd, desc="compute cer score:"):
        fd_path = os.path.join(parent_real_path, fd_name)
        lst_im = os.listdir(fd_path)
        for im_name in lst_im:
            real_im_path = os.path.join(fd_path, im_name)
            fake_im_path = os.path.join(parent_fake_path, fd_name, im_name)
            try:
                with open(real_im_path, 'r') as freal:
                    ocr_real = freal.readlines()[0][:-1]
                with open(fake_im_path, 'r') as ffake:
                    ocr_fake = ffake.readlines()[0][:-1]
            except:
                print(real_im_path)

            lst_text_real.append(ocr_real)
            lst_text_fake.append(ocr_fake)

    if cer_method == 'local':
        return cer_method_local(lst_text_real, lst_text_fake)
    elif cer_method == "cer-torch":
        return cer_method_torch(lst_text_real, lst_text_fake)
    elif cer_method == 'editdistance':
        return cer_method_editdistance(lst_text_real, lst_text_fake)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute useful evaluation metrics (CER, WER, SER, ...)')
    parser.add_argument(
        '-r', '--reference', type=str, metavar='REF', default=None,
        help='reference sentence or file')
    parser.add_argument(
        '-t', '--transcription', type=str, metavar='HYP', default=None,
        help='transcription sentence or file')
    parser.add_argument(
        '-i', '--input_source', type=str, choices=('-', 'str', 'file'),
        default='-', help=""""-" reads parallel sentences from the standard
        input, "str" interprets `-r' and `-t' as sentences, and "file"
        interprets `-r' and `-t' as two parallel files, with one sentence per
        line (default: -)""")
    parser.add_argument(
        '-s', '--separator', type=str, metavar='SEP', default='\t',
        help="""use this string to separate the reference and transcription
        when reading from the standard input (default: \\t)""")
    parser.add_argument(
        '-e', '--encoding', type=str, metavar='ENC', default='utf-8',
        help="""character encoding of the reference and transcription text
        (default: utf-8)""")
    args = parser.parse_args()

    if args.input_source != '-' and \
            (args.reference is None or args.transcription is None):
        logging.error('Expected reference and transcription sources')
        exit(1)

    ref, hyp = [], []
    if args.input_source == 'str':
        ref.append(args.reference.decode(args.encoding))
        hyp.append(args.transcription.decode(args.encoding))
    elif args.input_source == '-':
        line_n = 0
        for line in sys.stdin:
            line_n += 1
            line = line.rstrip('\n').rstrip('\r').decode(args.encoding)
            fields = line.split(args.separator)
            if len(fields) != 2:
                logging.warning(
                    'Line %d has %d fields but 2 were expected',
                    line_n, len(fields))
                continue
            ref.append(fields[0])
            hyp.append(fields[1])
    elif args.input_source == 'file':
        ref = load_file(args.reference, args.encoding)
        hyp = load_file(args.transcription, args.encoding)
        if len(ref) != len(hyp):
            logging.error(
                'The number of reference and transcription sentences does not '
                'match (%d vs. %d)', len(ref), len(hyp))
            exit(1)
    else:
        logging.error('INPUT FROM "%s" NOT IMPLEMENTED', args.input_source)
        exit(1)

    wer_s, wer_i, wer_d, wer_n = 0, 0, 0, 0
    cer_s, cer_i, cer_d, cer_n = 0, 0, 0, 0
    sen_err = 0
    for n in range(len(ref)):
        # update CER statistics
        _, (s, i, d) = levenshtein(ref[n], hyp[n])
        cer_s += s
        cer_i += i
        cer_d += d
        cer_n += len(ref[n])
        # update WER statistics
        _, (s, i, d) = levenshtein(ref[n].split(), hyp[n].split())
        wer_s += s
        wer_i += i
        wer_d += d
        wer_n += len(ref[n].split())
        # update SER statistics
        if s + i + d > 0:
            sen_err += 1

    if cer_n > 0:
        print(
            f'CER: {(100.0 * (cer_s + cer_i + cer_d)) / cer_n}, WER: {(100.0 * (wer_s + wer_i + wer_d)) / wer_n}, SER: {(100.0 * sen_err) / len(ref)}')
