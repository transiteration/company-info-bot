import os
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from graph import app

load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Sends welcome message when /start is used
    user = update.effective_user
    await update.message.reply_html(
        f"Hi {user.mention_html()}! I'm your company's information assistant.\n\n"
        "Ask me anything about our policies, projects, or history. For example:\n"
        "- What is our remote work policy?\n"
        "- Tell me about Project Alpha.\n"
        "- Who founded our company?"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Sends help message when /help is used
    await update.message.reply_text(
        "I can answer questions based on internal company documents. "
        "Just type your question and I'll do my best to find the answer for you."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Handles user messages and sends them to the RAG graph
    user_question = update.message.text
    chat_id = update.message.chat_id

    thinking_message = await context.bot.send_message(chat_id, "🧠 Thinking...")
    inputs = {"question": user_question, "transform_attempts": 0} 

    try:
        final_answer = ""
        final_state = app.invoke(inputs)
        final_answer = final_state.get("generation", "Sorry, I couldn't find an answer.")

        await context.bot.edit_message_text(
            text=final_answer,
            chat_id=chat_id,
            message_id=thinking_message.message_id
        )

    except Exception as e:
        print(f"An error occurred: {e}")
        await context.bot.edit_message_text(
            text="An error occurred while processing your request. Please try again.",
            chat_id=chat_id,
            message_id=thinking_message.message_id
        )

def main() -> None:
    # Starts the Telegram bot
    print("Starting bot...")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running. Press Ctrl-C to stop.")
    application.run_polling()

if __name__ == '__main__':
    main()