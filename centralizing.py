import pandas as pd

# file paths
chatgpt_path = r"D:\VSCODE\inn_ai\chatbot\chatgpt_conversation.xlsx"
gemini_path = r"D:\VSCODE\inn_ai\chatbot\Gemini_conversation.xlsx"
claude_path = r"D:\VSCODE\inn_ai\chatbot\cluade_conversation.csv"

# load datasets
chatgpt = pd.read_excel(chatgpt_path)
gemini = pd.read_excel(gemini_path)
claude = pd.read_csv(claude_path)

# rename columns
chatgpt = chatgpt.rename(columns={
    "Conversation Title":"title",
    "User Prompt":"prompt",
    "ChatGPT Response":"response"
})
chatgpt["source"] = "ChatGPT"

gemini = gemini.rename(columns={
    "Conversation Title":"title",
    "User Prompt":"prompt",
    "Gemini Response":"response"
})
gemini["source"] = "Gemini"

claude["title"] = "Claude conversation"

# select same columns
chatgpt = chatgpt[["source","title","prompt","response"]]
gemini = gemini[["source","title","prompt","response"]]
claude = claude[["source","title","prompt","response"]]

# merge datasets
central_data = pd.concat([chatgpt, gemini, claude])

# save centralized file
central_data.to_csv(r"D:\VSCODE\inn_ai\chatbot\centralized_conversations.csv", index=False)

print("Centralization completed!")