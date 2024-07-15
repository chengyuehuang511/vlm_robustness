import logging
from PIL import Image
import requests
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, PaliGemmaConfig
from lavis.common.registry import registry
from lavis.models.base_model import BaseModel

class PaliGemma_VQA(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.model = PaliGemmaForConditionalGeneration(config)
        self.processor = AutoProcessor.from_pretrained("google/PaliGemma-test-224px-hf")
        self.max_txt_len = config.max_txt_len

    def forward(self, samples):
        encoder_output, image_embeds = self.forward_encoder(samples)
        loss, decoder_output, decoder_targets = self.forward_decoder(
            samples=samples, encoder_out=encoder_output
        )

        return {
            "loss": loss,
            "intermediate_output": {
                "image_embeds": image_embeds,
                "encoder_output": encoder_output,
                "decoder_output": decoder_output,
                "decoder_labels": decoder_targets,
            }
        }

    def forward_encoder(self, samples):
        questions = samples["text_input"]
        questions = self.processor(
            questions,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        questions.input_ids[:, 0] = self.processor.cls_token_id
        samples.update({"tokenized_text": questions})

        image_embeds = self.model.vision_tower.forward_features(samples["image"])
        encoder_output = self.model.language_model.encoder(
            input_ids=samples["tokenized_text"].input_ids,
            attention_mask=samples["tokenized_text"].attention_mask,
            encoder_hidden_states=image_embeds
        )

        return encoder_output, image_embeds

    def forward_decoder(self, samples, encoder_out, **kwargs):
        answers = self.processor(
            samples["answer"], padding="longest", return_tensors="pt"
        ).to(self.device)
        answers.input_ids[:, 0] = self.processor.bos_token_id
        answer_targets = answers.input_ids.masked_fill(
            answers.input_ids == self.processor.pad_token_id, -100
        )

        question_states = []
        question_atts = []

        question = samples["tokenized_text"]
        question_output = encoder_out

        for b, n in enumerate(samples["n_answers"]):
            question_states += [question_output.last_hidden_state[b]] * n
            question_atts += [question.attention_mask[b]] * n

        question_states = torch.stack(question_states, dim=0)
        question_atts = torch.stack(question_atts, dim=0)

        answer_output = self.model.language_model.decoder(
            input_ids=answers.input_ids,
            attention_mask=answers.attention_mask,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            labels=answer_targets,
            return_dict=True,
            reduction="none",
        )

        loss = samples["weight"] * answer_output.loss
        bsz = samples["image"].size(0)

        loss = loss.sum() / bsz

        return loss, answer_output, answer_targets

# Function to process a single VQA example
def process_vqa_example(question, image_url, answer=None):
    # Load the image
    image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
    image_tensor = model.model.vision_tower.transform(image).unsqueeze(0)  # Transform and add batch dimension

    # Prepare the question
    question_tokenized = model.tokenizer(question, padding="longest", truncation=True, return_tensors="pt")

    inputs = {
        "image": image_tensor,
        "text_input": [question],
        "answer": [answer] if answer is not None else None,
        "weight": torch.tensor([1.0]) if answer is not None else None,
        "n_answers": torch.tensor([1]) if answer is not None else None
    }
    
    if answer is not None:
        inputs["answer"] = [answer]
        inputs["weight"] = torch.tensor([1.0])
        inputs["n_answers"] = torch.tensor([1])

    return inputs

# Example VQA v2 data
question = "Where is the cow standing?"
image_url = "https://huggingface.co/gv-hf/PaliGemma-test-224px-hf/resolve/main/cow_beach_1.png"
answer = "beach"  # This would be the ground truth answer in the dataset

# Initialize model
config = PaliGemmaConfig.from_pretrained("google/PaliGemma-test-224px-hf")
model = PaliGemma_VQA(config)

# Process a single example
inputs = process_vqa_example(question, image_url, answer)

# Forward pass
output = model(inputs)
loss = output["loss"]

print(f"Loss: {loss.item()}")
