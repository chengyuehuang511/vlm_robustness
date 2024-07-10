import logging
from PIL import Image
import requests
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, PaliGemmaConfig, PaliGemmaProcessor
from lavis.common.registry import registry
from lavis.models.base_model import BaseModel
from transformers import Trainer
import numpy as np

@registry.register_model("paligemma_vqa")
class PaliGemma_VQA(BaseModel):  # TODO
    """
    Paligemma VQA model.
    Supported model types:
        - paligemma-3b-ft-vqav2-448: fine-tuned model with VQAv2 dataset
        - paligemma-3b-pt-224: pre-trained model
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "paligemma-3b-ft-vqav2-448": "/nethome/chuang475/flash/projects/vlm_robustness/configs/models/paligemma_vqa/paligemma_ft_vqav2.yaml",
        "paligemma-3b-ft-vqav2-224": "/nethome/chuang475/flash/projects/vlm_robustness/configs/models/paligemma_vqa/paligemma_ft_vqav2_224.yaml",
        "paligemma-3b-pt-224": "/nethome/chuang475/flash/projects/vlm_robustness/configs/models/paligemma_vqa/paligemma_pt_224.yaml",
    }

    def __init__(
        self,
        model_id="google/paligemma-3b-ft-vqav2-448",  # paligemma-3b-ft-vqav2-448  paligemma-3b-pt-224
        dtype=torch.bfloat16,
        apply_lemmatizer=False,
    ):
        super().__init__()
        self.model_id = model_id
        print("model_id", model_id)
        self.dtype = dtype

        self.processor = AutoProcessor.from_pretrained(model_id)
        
        model_config = PaliGemmaConfig.from_pretrained(model_id)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            revision="bfloat16",
        )#.eval()  # TODO: check if eval is needed

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

    def forward(self, samples):
        # print("questions", questions)
        # print("answers", answers)
        # print("weight", samples["weight"])
        # print("n_answers", samples["n_answers"])
        
        # image_stack = []
        # questions_stack = []

        # assert len(samples["answer"]) == sum(samples["n_answers"])
        # assert len(samples["text_input_raw"]) == len(samples["n_answers"])
        # assert len(samples["image_raw"]) == len(samples["n_answers"])

        # for b, n in enumerate(samples["n_answers"]):
        #     # image
        #     image_stack += [samples["image_raw"][b]] * n
        #     # questions
        #     questions_stack += [samples["text_input_raw"][b]] * n

        # model_inputs = self.processor(text=questions_stack, images=image_stack, suffix=samples["answer"], return_tensors="pt", padding="longest").to(self.device)
        model_inputs = self.processor(text=samples["text_input_raw"], images=samples["image_raw"], suffix=samples["multiple_choice_answer"], return_tensors="pt", padding="longest").to(self.device)
        outputs = self.model(**model_inputs)
        loss = outputs.loss
        # print("loss: ", loss)
        
        return {"loss": loss}
    
    def predict_answers(
            self, 
            samples,
            num_beams=5,
            inference_method="generate",
            max_len=10,
            min_len=1,
            num_ans_candidates=128,
            answer_list=None,
            prompt="",
            length_penalty=-1,
            **kwargs
        ):
        # print("samples keys", samples.keys())
        image = samples["image_raw"]

        if isinstance(samples["text_input_raw"], str):
            samples["text_input_raw"] = [samples["text_input_raw"]]
        if prompt:
            text_input = [prompt.format(question) for question in samples["text_input_raw"]]
        else:
            text_input = samples["text_input_raw"]

        # print("image", image)
        # print("text_input", text_input)

        model_inputs = self.processor(text=text_input, images=image, return_tensors="pt", padding="longest").to(self.device)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
            # When the model generates a response, it appends the generated tokens to this input sequence.
            outputs = outputs[:, input_len:]
            output_text = self.processor.batch_decode(outputs, skip_special_tokens=True)
        
        if self._apply_lemmatizer:
            output_text = self._lemmatize(output_text)

        return output_text
    
    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        model_id = cfg.get("model_id", "google/paligemma-3b-ft-vqav2-448")  # paligemma-3b-ft-vqav2-448  paligemma-3b-pt-224
        dtype = cfg.get("dtype", torch.bfloat16)

        model = cls(
            model_id=model_id,
            dtype=dtype,
        )
        return model

if __name__ == "__main__":
    # Example usage:
    model_id = "google/paligemma-3b-ft-vqav2-448"
    device = "cuda:0"
    dtype = torch.bfloat16

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    # print("image", np.array(image))

    samples = {
        "image_raw": [image, image],
        "text_input_raw": ["caption es", "caption es: "],
    }
    with torch.inference_mode():
        model = PaliGemma_VQA(model_id=model_id, dtype=dtype).to(device)
        output = model.predict_answers(samples)
        print(output)
