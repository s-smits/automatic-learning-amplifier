# auto-local-document-finetuner
auto-local-document-finetuner is a Python project based on MLX that generates synthetic question-answer pairs from a given text corpus locally, finetunes a language model using the generated data, and deploys the finetuned model for inference. This project is particularly useful for creating domain-specific chatbots or question-answering systems in various fields. It is perfect for on-premise enterprise implementations.

## Features

- Generates specific, synthetic question-answer pairs from a large text corpus using LLaMA-3-8B with MLX or GGUF back-end
- Verifies the correctness of the generated question-answer pairs (optional)
- Analyzes the specificity of the generated questions and answers (optional)
- Supports local, OpenAI, OpenRouter, and Claude models for inference
- Automatically finetunes the LLaMA-3-8B model using the generated data with (Q)LoRA
- Converts the finetuned model to MLX or GGUF format for efficient inference
- Quantizes the model to reduce memory footprint without significant loss in perplexity
- Deploys the finetuned model for inference
- Compares the performance of the finetuned model with the original model


## Installation

### Using pip

1. Clone the repository:
   ```
   git clone https://github.com/s_smits/auto-local-document-finetuner.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Using Poetry

1. Clone the repository:
   ```
   git clone https://github.com/s_smits/auto-local-document-finetuner.git
   ```

2. Install Poetry if you haven't already:
   ```
   pip install poetry
   ```

3. Install the project dependencies:
   ```
   poetry install
   ```

3. Set up the necessary environment variables if using OpenAI, Claude or OpenRouter:
   - 'OPENAI_API_KEY': Your OpenAI API key
   - 'CLAUDE_API_KEY': Your Anthropic Claude API key
   - 'OPENROUTER_API_KEY': Your OpenRouter API key

## Usage

1. Prepare your text corpus in the `data/documents` folder.

2. Run the `main.py` script with the desired arguments:
   ```
   python main.py [--local] [--openrouter] [--claude] [--word_limit WORD_LIMIT] [--question_amount QUESTION_AMOUNT] [--focus {processes,knowledge,formulas}] [--images] [--optimize] [--add_summary {math,science,history,geography,english,art,music,education,computer science,drama}] [--verify] [--test_size TEST_SIZE] [--lora] [--qlora] [--gguf] [--compare]
   ```

   - `--local`, `--openrouter`, or `--claude`: Choose the inference method (default: `--local`)
   - `--word_limit`: Set the word limit for each chunk (default: 1000)
   - `--question_amount`: Set the number of question-answer pairs to generate (default: 5)
   - `--focus`: Choose the focus for generating questions (options: `processes`, `knowledge`, `formulas`)
   - `--images`: Generate captions for images in the documents (default: False)
   - `--optimize`: Activate optimization mode (default: False)
   - `--add_summary`: Add a short summary to each prompt from every 5 document files to give the model more context in order to generate better questions (options: 'general', `math`, `science`, `history`, `geography`, `english`, `art`, `music`, `education`, `computer science`, `drama`) (default: None)
   - `--verify`: Verify the generated questions and answers (default: False)
   - `--test_size`: Set the test size for splitting the data into training and validation sets (default: 0.1)
   - `--lora`: Finetune the model with LoRA (default: False)
   - `--qlora`: Finetune the model with QLoRA (default: True)
   - `--gguf`: Convert and infer the model with GGUF (default: False)
   - `--compare`: Compare the performance of the finetuned model with the non-finetuned model (default: True)

3. The script will generate synthetic question-answer pairs, finetune the language model, and deploy the finetuned model for inference and comparison with the original model.

## Folder Structure

- `data/`: Contains the text corpus and generated data
- `documents/`: Place your documents here (Supported file types: PDF, TXT, DOCX, PPTX, HTML)
- `data_prepared/`: Prepared data for finetuning
- `data_ft/`: Finetuning data (train and eval splits)
- `qa_json/`: Generated question-answer pairs in JSON format
- `logs/`: Contains log files
- `models/`: Contains the finetuned model and adapter file
- `src/`: Contains the source code files

## Why MLX?

MLX is an array framework for machine learning research on Apple silicon,
brought to you by Apple machine learning research.

Some key features of MLX include:

- **Familiar APIs**: MLX has a Python API that closely follows NumPy.  MLX
   also has fully featured C++, [C](https://github.com/ml-explore/mlx-c), and
   [Swift](https://github.com/ml-explore/mlx-swift/) APIs, which closely mirror
   the Python API.  MLX has higher-level packages like `mlx.nn` and
   `mlx.optimizers` with APIs that closely follow PyTorch to simplify building
   more complex models.

- **Composable function transformations**: MLX supports composable function
   transformations for automatic differentiation, automatic vectorization,
   and computation graph optimization.

- **Lazy computation**: Computations in MLX are lazy. Arrays are only
   materialized when needed.

- **Dynamic graph construction**: Computation graphs in MLX are constructed
   dynamically. Changing the shapes of function arguments does not trigger
   slow compilations, and debugging is simple and intuitive.

- **Multi-device**: Operations can run on any of the supported devices
   (currently the CPU and the GPU).

- **Unified memory**: A notable difference from MLX and other frameworks
   is the *unified memory model*. Arrays in MLX live in shared memory.
   Operations on MLX arrays can be performed on any of the supported
   device types without transferring data.

## Acknowledgement

I would like to send my many thanks to:

- The Apple Machine Learning Research team for the amazing MLX library.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [Apache 2.0](LICENSE).

## Acknowledgements

- [MLX](https://github.com/ml-explore/mlx) for the `mlx` package
- [Llama.cpp](https://github.com/ggerganov/llama.cpp) for better inference
- [OpenRouter](https://openrouter.ai/) and [Anthropic Claude](https://www.anthropic.com/) for alternative inference methods

## Disclaimer

This project has been primarily tested with the LLaMA-3-8B model. Using other models may result in poorly parsed JSON outputs. Please exercise caution and verify the generated data when using different models.
