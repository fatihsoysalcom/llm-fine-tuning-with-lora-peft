# LLM Fine-tuning with LoRA PEFT

This example demonstrates how to fine-tune a small Large Language Model (LLM) using Parameter-Efficient Fine-Tuning (PEFT) with LoRA. It showcases the process of loading a pre-trained model, preparing a small synthetic dataset, configuring LoRA, and training the model to adapt it to specific instructions. The example highlights how PEFT methods enable efficient fine-tuning, making it feasible on single-GPU setups.

## Language

`python`

## How to Run

1. Install necessary libraries: `pip install torch transformers peft datasets accelerate`
2. Run the script: `python main.py`

## Original Article

This example accompanies the Turkish article: [Tek Bir H200 GPU Droplet Üzerinde Ornith 9b LLM'i İnce Ayarlama: Maliyet, Gecikme ve Sunum Yükü Analizi](https://fatihsoysal.com/blog/tek-bir-h200-gpu-droplet-uzerinde-ornith-9b-llmi-ince-ayarlama-maliyet-gecikme-ve-sunum-yuku-analizi/).

## License

MIT — see [LICENSE](LICENSE).
