from PIL import Image
import os
if __name__ == "__main__":
    set_train, set_test = [], []
    for split in ["test", 'train']:
        image_src = f"data/IAM64-new/{split}"
        with open(f"data/IAM64_{split}.txt", "r") as file:
            with open(f"data/IAM64_{split}_add.txt", "w") as final:
                train_data = file.readlines()
                train_data = [i.strip().split(' ') for i in train_data]
                for i in train_data:
                    s_id = i[0].split(',')[0]
                    image_ori = i[0].split(',')[1]
                    image_name = image_ori + '.png'
                    transcription = i[1]
                    image = Image.open(os.path.join(image_src, s_id, image_name))
                    w, h = image.size
                    if split == "train":
                        set_train.append(w)
                    else:
                        set_test.append(w)
                    final.write(f"{s_id},{image_ori} {transcription} {h} {w}\n")
    print(sorted(list(set(set_train))))
    print(sorted(list(set(set_test))))

