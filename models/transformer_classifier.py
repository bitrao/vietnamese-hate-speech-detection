from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from data.custom_dataset import CustomDataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os

class HateSpeechDetector:
    def __init__(self, model_name='roberta-base', num_labels=3):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
    def tokenize_data(self, texts):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='weighted')
        }
    
    def train(self, train_data, val_data=None,
              batch_size=16, num_epochs=3, learning_rate=2e-5, output_dir="./transfomer_model/"):
        
        train_dataset = CustomDataset(train_data['text'], train_data['labels'], self.tokenizer)
        val_dataset =  CustomDataset(val_data['text'], val_data['labels'], self.tokenizer)  
        
        output_dir = f"{output_dir}/{self.model_name}"
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            eval_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="f1" if val_dataset else None,
            save_total_limit=2,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics if val_dataset else None
        )
        
        # Train the model
        trainer.train()
        
        # Save the best model
        self.save_model(output_dir)
    
    def predict(self, texts):
        # Tokenize inputs
        inputs = self.tokenize_data(texts)
        
        # Get predictions
        outputs = self.model(**inputs)
        predictions = np.argmax(outputs.logits.detach().numpy(), axis=1)
        
        return predictions
    
    def save_model(self, output_dir):
        """Save the model and tokenizer to a directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(output_dir)
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save model name for reference
        with open(os.path.join(output_dir, "model_name.txt"), "w") as f:
            f.write(self.model_name)
    
    @classmethod
    def load_model(cls, model_dir, tokenizer=None):
        """Load a saved model from a directory"""
        # Create a new instance
        instance = cls.__new__(cls)
        
        # Load the model name if available
        try:
            with open(os.path.join(model_dir, "model_name.txt"), "r") as f:
                instance.model_name = f.read().strip()
        except FileNotFoundError:
            instance.model_name = "unknown"
        
        # Load tokenizer and model
        instance.tokenizer = AutoTokenizer.from_pretrained( tokenizer if tokenizer else model_dir)
        instance.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        
        return instance