# from transformers import AutoModelForCausalLM, AutoTokenizer


# def get_conditional_probability(
#     model,
#     tokenizer,
#     prompt,
#     response):
#     """
#     The
#     """
#     model.eval()
#     text_with_chat_template = tokenizer.apply_chat_template(
#         prompt,
#         tokenize=False
#     )
#     input_tensor = tokenizer(text_with_chat_template, return_tensors="pt").to("cuda")

#     # for t in
#     ouput = model(input_tensor)


# if __name__ == "main":

#     # Defect Context
#     output = None

#     # Cooperate Context
