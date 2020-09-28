from tqdm.auto import tqdm
import os
import concurrent.futures

a = [
        "python bulk.py --cuda --model-path models/deep/finetuen_sopi_latest.pth --lm-path lm/full/full_libri_tvt.binary --decoder beam --workers 48 --alpha 0.47 --beta 0.27 --batch-size 4 --beam-width 512 --path /media/nas_mount/Sarthak/ijcai_acl/prompt_wise_16k_audios/prompt1/ --save /media/data_dump/hemant/16k_audios/prompt1/ --m1 0 --m2 1 --gpu 4",
        "python bulk.py --cuda --model-path models/deep/finetuen_sopi_latest.pth --lm-path lm/full/full_libri_tvt.binary --decoder beam --workers 48 --alpha 0.47 --beta 0.27 --batch-size 4 --beam-width 512 --path /media/nas_mount/Sarthak/ijcai_acl/prompt_wise_16k_audios/prompt2/ --save /media/data_dump/hemant/16k_audios/prompt2/ --m1 0 --m2 1 --gpu 4",
        "python bulk.py --cuda --model-path models/deep/finetuen_sopi_latest.pth --lm-path lm/full/full_libri_tvt.binary --decoder beam --workers 48 --alpha 0.47 --beta 0.27 --batch-size 4 --beam-width 512 --path /media/nas_mount/Sarthak/ijcai_acl/prompt_wise_16k_audios/prompt3/ --save /media/data_dump/hemant/16k_audios/prompt3/ --m1 0 --m2 1 --gpu 4",
        "python bulk.py --cuda --model-path models/deep/finetuen_sopi_latest.pth --lm-path lm/full/full_libri_tvt.binary --decoder beam --workers 48 --alpha 0.47 --beta 0.27 --batch-size 4 --beam-width 512 --path /media/nas_mount/Sarthak/ijcai_acl/prompt_wise_16k_audios/prompt4/ --save /media/data_dump/hemant/16k_audios/prompt4/ --m1 0 --m2 1 --gpu 4",
        "python bulk.py --cuda --model-path models/deep/finetuen_sopi_latest.pth --lm-path lm/full/full_libri_tvt.binary --decoder beam --workers 48 --alpha 0.47 --beta 0.27 --batch-size 4 --beam-width 512 --path /media/nas_mount/Sarthak/ijcai_acl/prompt_wise_16k_audios/prompt5/ --save /media/data_dump/hemant/16k_audios/prompt5/ --m1 0 --m2 1 --gpu 4",
        "python bulk.py --cuda --model-path models/deep/finetuen_sopi_latest.pth --lm-path lm/full/full_libri_tvt.binary --decoder beam --workers 48 --alpha 0.47 --beta 0.27 --batch-size 4 --beam-width 512 --path /media/nas_mount/Sarthak/ijcai_acl/prompt_wise_16k_audios/prompt6/ --save /media/data_dump/hemant/16k_audios/prompt6/ --m1 0 --m2 1 --gpu 4",
        "python bulk.py --cuda --model-path models/deep/finetuen_sopi_latest.pth --lm-path lm/full/full_libri_tvt.binary --decoder beam --workers 48 --alpha 0.47 --beta 0.27 --batch-size 4 --beam-width 512 --path /media/nas_mount/Sarthak/ijcai_acl/prompt_wise_16k_audios/prompt7/ --save /media/data_dump/hemant/16k_audios/prompt7/ --m1 0 --m2 1 --gpu 4",
        "python bulk.py --cuda --model-path models/deep/finetuen_sopi_latest.pth --lm-path lm/full/full_libri_tvt.binary --decoder beam --workers 48  --alpha 0.47 --beta 0.27 --batch-size 4 --beam-width 512 --path /media/nas_mount/Sarthak/ijcai_acl/prompt_wise_16k_audios/prompt8/ --save /media/data_dump/hemant/16k_audios/prompt8/ --m1 0 --m2 1 --gpu 4",
        "python bulk.py --cuda --model-path models/deep/finetuen_sopi_latest.pth --lm-path lm/full/full_libri_tvt.binary --decoder beam --workers 32 --alpha 0.47 --beta 0.27 --batch-size 3 --beam-width 512 --path /media/nas_mount/Sarthak/audios_16k_latest/ --save /media/nas_mount/hemant/16k_audios/prompt/ --m1 0.0 --m2 1--gpu 4"
    
]

def run(i):
    print(f"running: {i}")
    os.system(i)

# run(a[8])
with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
   results = [executor.submit(run, i) for i in a]
