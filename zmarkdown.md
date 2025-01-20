We appreciate the reviewer’s kind comments and the suggestion of adding more backbones and visualizations of the projection strength across the layers during training. We have included thorough answers in our response and are happy to engage further during the rebuttal period if clarification is needed.

**There are many public unimodal and multimodal foundation models, e.g., MAE, CLIP, BEiT3, LLaVA, etc. It is unclear why ResNet50 and PaliGemma are selected as foundation models. The ResNet50 pretrained in a supervised manner on ImageNet can hardly be deemed as foundation models. The PaliGemma is pretrained on a broad mixture of large-scale vision-language tasks. Whether the conclusion of this paper holds across other, more general foundation models, e.g., CLIP or LLaVA, is questionable.**

We adopt the MOCO-V3 ResNet50 as the backbone for the image classification task, following and comparing to previous work ([1], [2]). MOCO-V3 ResNet50 builds on the ResNet50 architecture but is pre-trained on ImageNet-1k in a self-supervised manner rather than supervised. For the multi-modal backbone, we use PaliGemma-3B, a recently released lightweight model by Google that achieves state-of-the-art performance on VQAv2.

For this rebuttal, we additionally include results for DomainNet with CLIP ViT-Base (Tab. 10), DomainNet-oVQA (Tab. 9), and VQA (Tab. 8) with LLaVA in the **Additional Experiments for Rebuttal Section** in Appendix. DiGraP consistently achieves the best performance across all experiments. Thank you for this suggestion, and we will include these results in the paper.

---

**Although DiGraP demonstrates improved performance in near OOD settings, its performance on far OOD tasks remains limited compared to other methods. The authors acknowledge this trade-off between ID/near OOD and far OOD robustness, but a deeper investigation into addressing this limitation would enhance the model’s versatility.**  
**Could additional techniques be incorporated to enhance DiGraP’s performance on far OOD datasets without sacrificing ID and near OOD performance?**

Thanks for pointing this out. Enhancing robustness across both near and far OOD settings is a challenging and underexplored problem. Most prior work, such as [1][2][3][4], focuses on general OOD robustness without explicitly differentiating between near and far OOD scenarios. Addressing DiGraP’s limitations on far OOD datasets while maintaining strong ID and near OOD performance requires a more detailed investigation into the dynamics of projection strength and its interaction with diverse distribution shifts. This remains a promising direction for future research, which could significantly enhance the versatility and robustness of DiGraP across a wider range of tasks.

---

**Could the authors provide more insights into how the projection strength parameter adapts dynamically across different layers and tasks, especially in multi-modal settings?**

We present a visualization of the variation in regularization strength ($\lambda$) across different layers over epochs in Fig. 9 of the **Additional Experiments for Rebuttal Section** in Appendix. The results show that the regularization strength evolves dynamically during training, starting small, increasing over iterations, and eventually converging. In the vision layers (blue), early layers tend to experience weaker regularization compared to later layers throughout the training process. Conversely, the language layers (orange) display a more uniform regularization strength, with comparable levels observed between early and later layers.

The weaker regularization in early vision layers likely allows them to preserve foundational low-level features, while stronger regularization in later layers encourages the model to focus on high-level semantic representations. Thank you for the suggestion; we will include these visualizations in the paper, hoping they inspire additional ideas for future work.

---

**The model’s gradient projection mechanism, while theoretically sound, lacks interpretability in how projection strength decisions impact specific instances.**

DiGraP is designed to balance pre-trained and fine-tuned trajectories by dynamically adjusting projection strength to optimize overall training directions. This adjustment can be interpreted layer-wise, showing how different layers contribute to balancing pre-trained knowledge and fine-tuning, and iteration-wise, revealing how projection strength evolves during training. However, it lacks interpretability at the instance level, as it focuses on global optimization across layers and iterations rather than tailoring projection strength to individual data instances.

---

**The figures in Appendix have too small font.**

Thanks for the suggestion! We will increase the font size and update the figures.


[1] Junjiao Tian, Xiaoliang Dai, Chih-Yao Ma, Zecheng He, Yen-Cheng Liu, and Zsolt Kira. Trainable Projected Gradient Method for Robust Fine-tuning, March 2023a.

[2] Junjiao Tian, Yen-Cheng Liu, James Seale Smith, and Zsolt Kira. Fast Trainable Projection for Robust Fine-Tuning, October 2023b.

[3] Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs Raphael Gontijo-Lopes, Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong, and Ludwig Schmidt. Robust fine-tuning of zero-shot models, June 2022

[4] Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, and Percy Liang. Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution, February 2022. 