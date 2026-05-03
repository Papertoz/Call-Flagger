import pandas as pd
import numpy as np
import random
import uuid
import os

# Ensure reproducibility
np.random.seed(42)
random.seed(42)

def generate_synthetic_data(num_samples=1000, output_path=None):
    """
    Generates a synthetic dataset for the Call Quality Auto-Flagger.
    Creates a highly imbalanced dataset where ~9% of calls have a ticket (has_ticket=1).
    Now includes realistic noise and overlap between classes to avoid perfect F1 scores.
    """
    print(f"Generating {num_samples} realistic synthetic call records...")

    data = []
    
    # Anomaly templates
    medical_phrases = [
        "I recommend taking ibuprofen for the pain.",
        "You should stop taking that medication immediately.",
        "Based on your symptoms, it sounds like diabetes.",
        "Let me prescribe something for your fever."
    ]
    
    angry_phrases = [
        "This is ridiculous, I want to speak to a manager now!",
        "You guys are completely useless.",
        "I'm going to sue your company.",
        "Cancel my account right now!"
    ]
    
    # Subtle negative phrases that don't trigger the regex but might be caught by embeddings
    subtle_negative_phrases = [
        "I really don't understand why this is so difficult.",
        "I've been waiting for months and nothing has happened.",
        "This is very disappointing service.",
        "I guess I'll just have to deal with it myself."
    ]
    
    normal_phrases = [
        "Thank you for calling support, how can I help you?",
        "Could you please verify your account number?",
        "I have processed your request, is there anything else?",
        "Have a great day, thank you for being a customer.",
        "Let me check the status of your order."
    ]
    
    for _ in range(num_samples):
        call_id = str(uuid.uuid4())
        
        # Determine if this call will be flagged (9% chance)
        has_ticket = 1 if random.random() < 0.09 else 0
        
        # Generate base features with normal distributions
        duration = int(np.random.normal(loc=400, scale=150))
        duration = max(30, min(3600, duration))
        
        num_questions_asked = random.randint(1, 5)
        
        qa_mismatch = 0
        wrong_number_flag = 0
        medical_advice_flag = 0
        transcript_sentences = random.sample(normal_phrases, k=random.randint(1, 3))
        
        # NOISE INJECTION: 
        # Sometimes normal calls have "angry" people but the agent resolves it (no ticket needed)
        if has_ticket == 0 and random.random() < 0.15:
            transcript_sentences.append(random.choice(angry_phrases))
            transcript_sentences.append("I am glad we could resolve that for you.")
            
        # Sometimes normal calls are just very short
        if has_ticket == 0 and random.random() < 0.05:
            duration = random.randint(15, 45)
            num_questions_asked = 0
            
        # FLAG INJECTION:
        if has_ticket == 1:
            anomaly_type = random.choice(['medical', 'angry', 'mismatch', 'wrong_number', 'short_duration', 'subtle_issue'])
            
            if anomaly_type == 'medical':
                medical_advice_flag = 1
                transcript_sentences.append(random.choice(medical_phrases))
            elif anomaly_type == 'angry':
                transcript_sentences.append(random.choice(angry_phrases))
            elif anomaly_type == 'mismatch':
                qa_mismatch = 1
            elif anomaly_type == 'wrong_number':
                wrong_number_flag = 1
                transcript_sentences.append("Sorry, I think I dialed the wrong number.")
            elif anomaly_type == 'short_duration':
                duration = random.randint(5, 20)
                num_questions_asked = 0
            elif anomaly_type == 'subtle_issue':
                # No hardcoded keyword flags triggered, only text embeddings will catch this
                transcript_sentences.append(random.choice(subtle_negative_phrases))
                
        # Shuffle transcript sentences and join
        random.shuffle(transcript_sentences)
        transcript = " ".join(transcript_sentences)
        
        data.append({
            'call_id': call_id,
            'duration': duration,
            'num_questions_asked': num_questions_asked,
            'qa_mismatch': qa_mismatch,
            'wrong_number_flag': wrong_number_flag,
            'medical_advice_flag': medical_advice_flag,
            'transcript': transcript,
            'has_ticket': has_ticket
        })
        
    df = pd.DataFrame(data)
    
    print(f"Generated data with shape {df.shape}")
    print(f"Target distribution:\n{df['has_ticket'].value_counts(normalize=True)}")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
        
    return df

if __name__ == "__main__":
    generate_synthetic_data(num_samples=2000, output_path="data/raw/synthetic_full.csv")
