import logging
from PIL import Image
import requests
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, PaliGemmaConfig, PaliGemmaProcessor, BitsAndBytesConfig
from lavis.common.registry import registry
from lavis.models.base_model import BaseModel
from lavis.common.utils import get_abs_path, is_url, download_cached_file
import numpy as np
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import contextlib
import copy
from tasks.vqa_task_utils import QAOutput

from llm_adapters.peft.src.peft import (  # noqa: E402
    BottleneckConfig,
    PrefixTuningConfig,
    get_peft_model as get_adapter_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

@registry.register_model("paligemma_vqa")
class PaliGemma_VQA(BaseModel):  # TODO
    """
    Paligemma VQA model.
    Supported model types:
        - paligemma-3b-ft-vqav2-448: fine-tuned model with VQAv2 dataset
        - paligemma-3b-pt-224: pre-trained model
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "paligemma-3b-ft-vqav2-448": "/coc/pskynet4/chuang475/projects/vlm_robustness/configs/models/paligemma_vqa/paligemma_ft_vqav2.yaml",
        "paligemma-3b-ft-vqav2-224": "/coc/pskynet4/chuang475/projects/vlm_robustness/configs/models/paligemma_vqa/paligemma_ft_vqav2_224.yaml",
        "paligemma-3b-pt-224": "/coc/pskynet4/chuang475/projects/vlm_robustness/configs/models/paligemma_vqa/paligemma_pt_224.yaml",
    }

    def __init__(
        self,
        model_id="google/paligemma-3b-pt-224",  # paligemma-3b-ft-vqav2-224  paligemma-3b-pt-224
        dtype=torch.bfloat16,
        apply_lemmatizer=False,
    ):
        super().__init__()

        self.model_id = model_id
        print("model_id", model_id)
        self.dtype = dtype

        self.processor = AutoProcessor.from_pretrained(model_id)
        
        model_config = PaliGemmaConfig.from_pretrained(model_id)

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            revision="bfloat16",
            # load_in_8bit=True,
            # quantization_config=quantization_config,
        )
        self.config = self.model.config

        print(self.config)
        # print('language_model.lm_head.weight', self.model.state_dict()['language_model.lm_head.weight'])

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None
    
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def forward(self, samples, **kwargs):
        # print("questions: ", samples["text_input_raw"])
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
        # model_inputs = self.processor(text=samples["text_input_raw"], images=samples["image_raw"], suffix=samples["multiple_choice_answer"], return_tensors="pt", padding="longest").to(self.device)
        # if samples["text_input_raw"] == ['Is there a light on?']:
        #     # print("model_inputs", model_inputs)
        #     # save json file of model_inputs
        #     # import json
        #     # with open("model_inputs.json", "w") as f:
        #     #     json.dump(model_inputs, f)

        #     # save model.state_dict() into a file
        #     torch.save(self.model.state_dict(), "model_state_dict.pth")

        #     # save image_raw, text_input_raw and multiple_choice_answer into 3 files
        #     # import pickle
        #     # with open("image_raw.pkl", "wb") as f:
        #     #     pickle.dump(samples["image_raw"], f)
        #     # with open("text_input_raw.pkl", "wb") as f:
        #     #     pickle.dump(samples["text_input_raw"], f)
        #     # with open("multiple_choice_answer.pkl", "wb") as f:
        #     #     pickle.dump(samples["multiple_choice_answer"], f)
            
        # print the trainable parameters require_grad==True
        # print("Trainable parameters: ")
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad == False:
        #         print(name)
        # for name, param in self.model.named_parameters():
        #     print(f"{name} requires_grad: {param.requires_grad}")

        # print('state_dict: ', self.model.state_dict())
        # print('language_model.lm_head.weight: ', self.model.state_dict()['language_model.lm_head.weight'])
        # print(id(self.model))
        # print("model_inputs", model_inputs)

        # # Monitoring gradients
        # for name, param in self.model.named_parameters():
        #     param.requires_grad_(True)
        #     print(f"Gradients for {name}: {param.grad}")
        #     print(f"If Leaf: {param.is_leaf}")
        
        # with self.maybe_autocast():
        # print("Paligemma receiving the following kwargs", kwargs)
        # print("Paligemma recieves the following samples", len(samples))

        model_inputs = self.processor(text=samples["text_input_raw"], images=samples["image_raw"], suffix=samples["multiple_choice_answer"], return_tensors="pt", padding="longest").to(self.dtype).to(self.device)
        # print("model_inputs", model_inputs)
        outputs = self.model(**model_inputs)
        # print("outputs", outputs)
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

        # with self.maybe_autocast():
        model_inputs = self.processor(text=text_input, images=image, return_tensors="pt", padding="longest").to(self.dtype).to(self.device)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
            # When the model generates a response, it appends the generated tokens to this input sequence.
            outputs = outputs[:, input_len:]
            output_text = self.processor.batch_decode(outputs, skip_special_tokens=True)
        
        if self._apply_lemmatizer:
            output_text = self._lemmatize(output_text)

        # print("output_text", output_text)

        if ("return_dict" in kwargs) and kwargs["return_dict"]:
            return QAOutput(answer=output_text)
        else:
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
        model_id = cfg.get("model_id", "google/paligemma-3b-pt-224")  # paligemma-3b-ft-vqav2-224  paligemma-3b-pt-448
        dtype = cfg.get("dtype", torch.bfloat16)

        model = cls(
            model_id=model_id,
            dtype=dtype,
        )

        load_finetuned = cfg.get("load_finetuned", False)

        # LoRA
        use_lora = int(cfg.get("use_lora", 0))
        lora_alpha = int(cfg.get("lora_alpha", 16))
        lora_rank = int(cfg.get("lora_rank", 4))
        target_modules = cfg.get("target_modules", "q_proj k_proj v_proj o_proj").split()

        if use_lora == 1:
            lora_config = LoraConfig(
                r=lora_rank,  # 4
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                bias="none",
                target_modules=target_modules,  # ['q_proj', 'k_proj', 'v_proj', 'o_proj'],  # qformer, qkv
            )
            
            logging.info(lora_config)
            # model = prepare_model_for_kbit_training(model)
            
            model = get_peft_model(model, lora_config)
            logging.info(model.print_trainable_parameters())
            # print(model) 
            # raise Exception("just testing")
        
        # print("model: ", model)

        # Adapter
        use_adapter = int(cfg.get("use_adapter", 0))
        use_parallel_adapter = int(cfg.get("use_parallel_adapter", 0))

        if use_adapter == 1 : 
            print("Use adapter")
            non_linearity=cfg.get("non_linearity", "tanh")
            bottleneck_size=int(cfg.get("bottleneck_size", 256))
            target_modules = cfg.get("target_modules", "q_proj k_proj v_proj o_proj").split()
            scaling = float(cfg.get("scaling", 1.0))
            adapter_dropout = float(cfg.get("adapter_dropout", 0.1))
            use_para_adapter = False
            if use_parallel_adapter == 1 : 
                use_para_adapter = True 
                print("use para_adapter")
                
            adapter_config = BottleneckConfig(
                        bottleneck_size=bottleneck_size,
                        non_linearity=non_linearity,
                        adapter_dropout=adapter_dropout,
                        use_parallel_adapter=use_para_adapter,
                        use_adapterp=False,
                        target_modules=target_modules,
                        scaling=scaling,
                        bias="none",
                        task_type="CAUSAL_LM",
                    )

            logging.info(adapter_config)
            print("prev dev", model.device)
            model = get_adapter_peft_model(model, adapter_config)
            
            logging.info(model.print_trainable_parameters())
            print(model)
        

        # Linear Probe
        linear_probe = int(cfg.get("linear_probe", 0))
        if linear_probe == 1:
            assert use_lora == 0, "Linear probe and LoRA cannot be used together"
            # only tune "lm_head" layer
            # for name, param in model.named_parameters():
            #     # if "lm_head" not in name:
            #     #     param.requires_grad_(False)
            #     # else:
            #     #     param.requires_grad_(True)
            #     print(f"{name} requires_grad: {param.requires_grad}")
            for name, module in model.named_modules():
                if "lm_head" not in name:
                    for param in module.parameters():
                        param.requires_grad_(False)
                        print(f"{name} requires_grad: {param.requires_grad}")
                else:
                    for param in module.parameters():
                        param.requires_grad_(True)
                        print(f"{name} requires_grad: {param.requires_grad}")
            logging.info("Linear probe: only tune 'lm_head' layer")
        
        # WiSE
        wise = int(cfg.get("wise", 0))
        if wise == 1:
            assert load_finetuned, "WiSE requires load_finetuned=True"
            w0 = {key: value.to('cpu') for key, value in model.state_dict().items()}
            w0 = copy.deepcopy(w0)
        
        if load_finetuned:
            model.load_checkpoint_from_config(cfg)
            # if use_adapter != 1 : 
            #     model.load_checkpoint_from_config(cfg)

            # else : #adapters 
            #     url_or_filename = cfg.get("finetuned", None)
            #     if os.path.isfile(cfg):
            #         checkpoint_name = torch.load(url_or_filename, map_location="cpu")
            #         adapters_weights = torch.load(checkpoint_name)
            #         model = set_peft_model_state_dict(model, adapters_weights)
        
        if wise == 1:
            w1 = {key: value.to('cpu') for key, value in model.state_dict().items()}
            # alpha * w0 + (1 - alpha) * w1
            alpha = 0.5
            wise = {key: (alpha * w0[key] + (1 - alpha) * w1[key]).to(model.device) for key in w1.keys()}
            model.load_state_dict(wise)
            logging.info("WiSE: load finetuned model and apply WiSE")


        print("Final Model before runner", model)
        # print("and it's device", model.device)
        return model
    
    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location=self.device)
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location=self.device)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        for key in list(state_dict.keys()):
            start_key = "base_model.model.model."
            if key.startswith(start_key):
                state_dict[key[len(start_key):]] = state_dict.pop(key)
        
        # Load the current state_dict of the model
        current_state_dict = self.model.state_dict()

        # Update the current state_dict with the new parameters
        for key in state_dict.keys():
            assert key in current_state_dict, f"key {key} not in current_state_dict"
            current_state_dict[key] = state_dict[key]

        # Load the updated state_dict back into the model
        self.model.load_state_dict(current_state_dict)
        logging.info("load pretrained checkpoint from %s" % url_or_filename)

    def load_checkpoint(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location=self.device)
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location=self.device)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        for key in list(state_dict.keys()):
            start_key = "base_model.model.model."
            start_key_2 = "model."
            if key.startswith(start_key):
                state_dict[key[len(start_key):]] = state_dict.pop(key)
            elif key.startswith(start_key_2):
                state_dict[key[len(start_key_2):]] = state_dict.pop(key)
        
        # Load the current state_dict of the model
        current_state_dict = self.model.state_dict()
        # print("address of current_state_dict", id(current_state_dict))
        # print("address of model.state_dict()", id(self.model.state_dict()))
        # print("current_state_dict", current_state_dict.keys())
        # print("state_dict", state_dict.keys())

        # Update the current state_dict with the new parameters
        print("Checkpoint state dict keys ", state_dict.keys()) 
        print("Current state dict keys", current_state_dict.keys())
        for key in state_dict.keys():
            assert key in current_state_dict, f"key {key} not in current_state_dict"
            current_state_dict[key] = state_dict[key]

        # Load the updated state_dict back into the model
        self.model.load_state_dict(current_state_dict)
        logging.info("load checkpoint from %s" % url_or_filename)


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
