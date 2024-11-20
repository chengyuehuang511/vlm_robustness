import json

def robust_ft():
    result_folder = "/coc/pskynet4/bmaneech3/vlm_robustness/result_output/contextual_ood/results_ood/"
    methods = ["pt", "fft", "vanilla_ft", "linear_prob", "lp_ft", "ftp", "spd"]
    files = ["pt_emb_ood_score_dict.json", "fft_ood_score_dict.json", "lora_ood_score_dict.json", "lp_ood_score_dict.json", "lpft_ood_score_dict.json", "ftp_ood_score_dict.json", "spd_ood_score_dict.json"]
    concepts = ["img_final", "joint", "ques_ft"]
    modalities = ["v|q,a", "q|v,a", "v,q,a"]
    datasets = ["coco_vqa_raw_val", "coco_iv-vqa_val", "coco_cv-vqa_val", "coco_vqa_rephrasings_val", "coco_vqa_cp_val", "coco_vqa_ce_val", "coco_advqa_val", "textvqa_val", "vizwiz_val", "coco_okvqa_val"]

    for i in range(len(concepts)):
        modality = modalities[i]
        concept = concepts[i]
        for j in range(len(methods)):
            method = methods[j]
            method_name = method.replace("_", "$\_$")
            file = files[j]
            js = json.load(open(result_folder + file, "r"))
            
            near_ood_mean = 0
            far_ood_mean = 0
            ood_mean = 0
            for i in range(len(datasets)):
                if i == 0:
                    continue
                elif i < 7:
                    near_ood_mean += -js[datasets[i]][concept]
                    ood_mean += -js[datasets[i]][concept]
                else:
                    far_ood_mean += -js[datasets[i]][concept]
                    ood_mean += -js[datasets[i]][concept]
            near_ood_mean /= 6
            far_ood_mean /= 3
            ood_mean /= 9

            latex = f"""
            $f_{{\\text{{{method_name}}}}}({modality})$ &               
                \\multicolumn{{1}}{{c|}}{{{-js[datasets[0]][concept]:.2f}}} &
                {-js[datasets[1]][concept]:.2f} & 
                \\multicolumn{{1}}{{c|}}{{{-js[datasets[2]][concept]:.2f}}} &
                \\multicolumn{{1}}{{c|}}{{{-js[datasets[3]][concept]:.2f}}} &
                \\multicolumn{{1}}{{c|}}{{{-js[datasets[4]][concept]:.2f}}} &
                \\multicolumn{{1}}{{c|}}{{{-js[datasets[5]][concept]:.2f}}} &
                \\multicolumn{{1}}{{c|}}{{{-js[datasets[6]][concept]:.2f}}} &
                {near_ood_mean:.2f}&
                \\multicolumn{{1}}{{c}}{{{-js[datasets[7]][concept]:.2f}}} &
                \\multicolumn{{1}}{{c}}{{{-js[datasets[8]][concept]:.2f}}} &
                \\multicolumn{{1}}{{c|}}{{{-js[datasets[9]][concept]:.2f}}}  &
                \\multicolumn{{1}}{{c|}}{{{far_ood_mean:.2f}}} &
                {ood_mean:.2f}
                \\\\
            """
            print(latex)
            # output the latex code to a file
            with open(f"latex.txt", "a") as f:
                f.write(latex)
        
        with open(f"latex.txt", "a") as f:
            f.write("\hline")


def uni_modal():
    files = ["vit_ood_score_dict.json", "uni_question_ood_score_dict.json"]
    concepts = ["uni_image", "question"]

    result_folder = "/coc/pskynet4/bmaneech3/vlm_robustness/result_output/contextual_ood/results_ood/"

    modalities = ["v", "q"]
    datasets = ["coco_vqa_raw_val", "coco_iv-vqa_val", "coco_cv-vqa_val", "coco_vqa_rephrasings_val", "coco_vqa_cp_val", "coco_vqa_ce_val", "coco_advqa_val", "textvqa_val", "vizwiz_val", "coco_okvqa_val"]

    for i in range(len(concepts)):
        concept = concepts[i]
        file = files[i]
        modality = modalities[i]
        js = json.load(open(result_folder + file, "r"))

        near_ood_mean = 0
        far_ood_mean = 0
        ood_mean = 0
        for i in range(len(datasets)):
            if i == 0:
                continue
            elif i < 7:
                near_ood_mean += -js[datasets[i]][concept]
                ood_mean += -js[datasets[i]][concept]
            else:
                far_ood_mean += -js[datasets[i]][concept]
                ood_mean += -js[datasets[i]][concept]
        near_ood_mean /= 6
        far_ood_mean /= 3
        ood_mean /= 9

        latex = f"""
            $f({modality})$ &
                                          
                \\multicolumn{{1}}{{c|}}{{{-js[datasets[0]][concept]:.2f}}} &
                {-js[datasets[1]][concept]:.2f} & 
                \\multicolumn{{1}}{{c|}}{{{-js[datasets[2]][concept]:.2f}}} &
                \\multicolumn{{1}}{{c|}}{{{-js[datasets[3]][concept]:.2f}}} &
                \\multicolumn{{1}}{{c|}}{{{-js[datasets[4]][concept]:.2f}}} &
                \\multicolumn{{1}}{{c|}}{{{-js[datasets[5]][concept]:.2f}}} &
                \\multicolumn{{1}}{{c|}}{{{-js[datasets[6]][concept]:.2f}}} &
                {near_ood_mean:.2f}&
                \\multicolumn{{1}}{{c}}{{{-js[datasets[7]][concept]:.2f}}} &
                \\multicolumn{{1}}{{c}}{{{-js[datasets[8]][concept]:.2f}}} &
                \\multicolumn{{1}}{{c|}}{{{-js[datasets[9]][concept]:.2f}}}  &
                \\multicolumn{{1}}{{c|}}{{{far_ood_mean:.2f}}} &
                {ood_mean:.2f}
                \\\\
            """
        
        print(latex)
        # output the latex code to a file
        with open(f"latex_unimodal.txt", "a") as f:
            f.write(latex)


if __name__ == "__main__":
    robust_ft()
    # uni_modal()