from openai import OpenAI

client = OpenAI(api_key='')
import os

# Set your OpenAI API key
def summarize_text(content):
    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text into one sentence."},
            {"role": "user", "content": content}
        ],
        max_tokens=150,
        temperature=0.5)
        summary = response.choices[0].message.content
        return summary.strip()
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return None

def process_files(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, f'summary_{filename}')

            with open(input_path, 'r', encoding='utf-8') as file:
                content = file.read()

            summary = summarize_text(content)

            if summary:
                with open(output_path, 'w', encoding='utf-8') as file:
                    file.write(summary)

            print(f"Processed {filename}, summary saved to {output_path}")

if __name__ == "__main__":
    input_directory = '/data/chaoyi_he/Video_gen/data/train_data/labels'  # Replace with your input directory path
    output_directory = '/data/chaoyi_he/Video_gen/data/train_data/summaries'  # Replace with your desired output directory path
    process_files(input_directory, output_directory)
