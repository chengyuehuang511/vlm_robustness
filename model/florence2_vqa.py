import logging
from PIL import Image
import requests
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForVision2Seq
from lavis.common.registry import registry
from lavis.models.base_model import BaseModel
from transformers import Trainer
import numpy as np

@registry.register_model("florence2_vqa")
class Florence2_VQA(BaseModel):
    """
    Florence-2 VQA model.
    Supported model types:
        - Florence-2-large-ft: fine-tuned model with a collection of datasets
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "Florence-2-large-ft": "/nethome/chuang475/flash/projects/vlm_robustness/configs/models/florence2_vqa/florence2_ft_vqav2.yaml"
    }

    def __init__(
        self,
        model_id="microsoft/Florence-2-large-ft",
        dtype=torch.bfloat16,
        apply_lemmatizer=False,
    ):
        super().__init__()
        self.model_id = model_id
        self.dtype = dtype

        self.processor = AutoProcessor.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            # revision='refs/pr/6',
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            # revision='refs/pr/6',
        )

        # # Load the model and processor
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     "microsoft/Florence-2-base-ft", trust_remote_code=True, revision="refs/pr/6"
        # )
        # self.processor = AutoProcessor.from_pretrained(
        #     "microsoft/Florence-2-base-ft", trust_remote_code=True, revision="refs/pr/6"
        # )

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

    def forward(self, samples):
        image = samples["image"]
        questions = samples["text_input"]
        answers = samples["answer"]

        model_inputs = self.processor(text=questions, images=image, suffix=answers, return_tensors="pt").to(self.device)
        outputs = self.model(**model_inputs)
        loss = outputs.loss
        
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
            text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input_raw"]

        # print("image", image)
        # print("text_input", text_input)

        model_inputs = self.processor(text=text_input, images=image, return_tensors="pt", padding="longest").to(self.device)
        # input_len = model_inputs["input_ids"].shape[-1]
        # print("model_inputs", model_inputs.keys())

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=model_inputs['input_ids'],
                pixel_values=model_inputs['pixel_values'], 
                # attention_mask=model_inputs['attention_mask'],
                max_new_tokens=100, 
                early_stopping=False,
                do_sample=False,
            )
            # When the model generates a response, it appends the generated tokens to this input sequence.
            # outputs = outputs[:, input_len:]
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
        model_id = cfg.get("model_id", "microsoft/Florence-2-large-ft")
        dtype = cfg.get("dtype", torch.bfloat16)

        model = cls(
            model_id=model_id,
            dtype=dtype,
        )
        return model

if __name__ == "__main__":
    # Example usage:
    model_id = "microsoft/Florence-2-large-ft"
    device = "cuda:0"
    dtype = torch.bfloat16

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    # print("image", np.array(image))

    samples = {
        "image_raw": [image, image],
        "text_input_raw": ["caption es", "What is the color of this car"],
    }
    with torch.inference_mode():
        model = Florence2_VQA(model_id=model_id, dtype=dtype).to(device)
        output = model.predict_answers(samples)
        print(output)
