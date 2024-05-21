def llamacpp_model_url(model_name: str) -> str:
    """
    Returns the URL for the specified Llama model.

    Args:
        model_name (str): The name of the Llama model.

    Returns:
        str: The URL of the model file.
    """
    if model_name == "llama2-13b":
        return "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"
    elif model_name == "llama2-7b":
        return "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
    elif model_name == "llama3-8b":
        return "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
    return "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"
