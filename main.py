from absl import app
from absl import flags
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from text_dataset import TextDataSet
from transformer_model import TransformerModel
from train_utils import train, generate_text

FLAGS = flags.FLAGS
flags.DEFINE_string('data_path', './test.csv', 'Path to the CSV dataset')
flags.DEFINE_integer('max_length', 128, 'Maximum sequence length')
flags.DEFINE_integer('batch_size', 4, 'Batch size for training')
flags.DEFINE_integer('epochs', 5, 'Number of training epochs')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training')
flags.DEFINE_string('start_text', 'Hello', 'Starting text for generation')
flags.DEFINE_integer('max_gen_length', 50, 'Maximum length of generated text')


def main(argv):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_ds = TextDataSet(FLAGS.data_path, tokenizer, FLAGS.max_length)
    train_dl = DataLoader(
        dataset=train_ds, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True)

    model = TransformerModel(v_size=tokenizer.vocab_size, d_model=512,
                             n_heads=8, num_layers=6, max_seq_length=FLAGS.max_length)

    train(model, train_dl, FLAGS.epochs,
          tokenizer.vocab_size, FLAGS.learning_rate)

    generated_text = generate_text(
        model, tokenizer, FLAGS.start_text, FLAGS.max_gen_length)
    print(f"Generated text: {generated_text}")


if __name__ == '__main__':
    app.run(main)
