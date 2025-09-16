"""
gradio_ui.py
This file creates the gradio interface for the ai tutor.
It uses the orchestrator.py file to handle the logic for the model calls and database management.


The parts of code not written by me are referenced from the following sources:
- gradio documentation to create the interface from https://www.gradio.app/docs/gradio/interface
- the css instructions from the gradio documentation 
  to implement the rtl styling from https://www.gradio.app/guides/custom-CSS-and-JS
- rtl sytling code from https://medium.com/techradiant/quick-guideline-for-rtl-ui-2da60615b655

"""

import gradio as gr
from typing import Dict, Any, List, Optional
from orchestrator import Orchestrator
from session_manager import SessionManager


# Translation dictionaries
TRANSLATIONS = {
    "en": {
        "title": "### Offline AI Tutor",
        "new_chat": "New Chat",
        "chats": "Chats",
        "delete_chat": "Delete Current Chat",
        "language": "Language",
        "speak_replies": "Speak replies",
        "tutor_speech": "Tutor Speech",
        "ai_tutor": "AI Tutor",
        "input_placeholder": "Type your question or add a file...",
        "send": "Send",
        "upload_image": "Upload Image",
        "record_audio": "Record or Upload Audio",
        "english": "English",
        "urdu": "Urdu"
    },
    "ur": {
        "title": "### آف لائن اے آئی ٹیوٹر",
        "new_chat": "نئی چیٹ",
        "chats": "گفتگو",
        "delete_chat": "موجودہ چیٹ حذف کریں",
        "language": "زبان",
        "speak_replies": "جوابات بول کر سنین",
        "tutor_speech": "ٹیوٹر کی آواز",
        "ai_tutor": "اے آئی ٹیوٹر",
        "input_placeholder": "اپنا سوال ٹائپ کریں یا کوئی فائل شامل کریں۔۔۔",
        "send": "بھیجیں",
        "upload_image": "تصویر اپ لوڈ کریں",
        "record_audio": "آواز ریکارڈ یا اپ لوڈ کریں",
        "english": "انگریزی",
        "urdu": "اردو"
    }
}

# Create an I18n instance with translations for English and Urdu
i18n = gr.I18n(**TRANSLATIONS)


def create_gradio_interface():
    """Create and configure the Gradio interface."""
    
    # Initialize the main orchestrator and session manager
    orchestrator = Orchestrator()
    session_manager = SessionManager()

    css = """
    /* RTL styling helper classes */
    .rtl {
        direction: rtl !important;
        text-align: right !important;
        unicode-bidi: embed !important;
    }

    /* Urdu font helper class */
    .urdu-text {
        font-family: 'Noto Nastaliq Urdu', 'Arial Unicode MS', sans-serif;
        font-size: 1.3em;
    }
    """
    
    with gr.Blocks(title="AI Tutor", theme=gr.themes.Default(), css=css) as demo:
        gr.Markdown(TRANSLATIONS["en"]["title"])

        with gr.Row():
            # Sidebar
            with gr.Column(scale=1, min_width=260):
                new_chat_btn = gr.Button(TRANSLATIONS["en"]["new_chat"], variant="primary")

                # Get the initial chat list from the database
                chat_choices = session_manager.list_session_choices()

                # chat_choices are lists [label, value]
                initial_value = chat_choices[0][1] 

                # Create the chat list
                chat_radio = gr.Radio(
                    choices=chat_choices,
                    value=initial_value,
                    label=TRANSLATIONS["en"]["chats"],
                    interactive=True,
                )

                # Create the delete button
                delete_btn = gr.Button(TRANSLATIONS["en"]["delete_chat"])

                # Create the language selector
                lang_selector = gr.Dropdown(
                    choices=[(TRANSLATIONS["en"]["english"], "english"), (TRANSLATIONS["en"]["urdu"], "urdu")],
                    value="english",
                    label=TRANSLATIONS["en"]["language"],
                    interactive=True,
                )

                # Create the speak toggle
                speak_toggle = gr.Checkbox(False, label=TRANSLATIONS["en"]["speak_replies"])

                # Create the tts audio interface
                tts_out = gr.Audio(label=TRANSLATIONS["en"]["tutor_speech"], interactive=False)

            # Main area
            with gr.Column(scale=4):
                
                # Get the initial session ID from the initial chat list
                initial_sid = initial_value  

                # Create the chatbot interface
                chatbot = gr.Chatbot(
                    value=session_manager.get_history(initial_sid, translator=orchestrator.translator),
                    elem_id="chatbot",
                    label=TRANSLATIONS["en"]["ai_tutor"],
                    height=600,
                    show_copy_button=False,
                    type="messages",
                )

                with gr.Row():
                    with gr.Column(scale=6):
                        # Create the text input interface
                        txt_input = gr.Textbox(
                            placeholder=TRANSLATIONS["en"]["input_placeholder"],
                            lines=2,
                            show_label=False,
                        )
                    with gr.Column(scale=1):
                        # Create the send button
                        send_btn = gr.Button(TRANSLATIONS["en"]["send"], variant="primary")

                with gr.Row():
                    # Create the image input interface
                    img_input = gr.Image(
                        type="filepath",
                        label=TRANSLATIONS["en"]["upload_image"],
                        sources=["upload", "webcam", "clipboard"],
                        scale=1,
                    )
                    # Create the audio input interface
                    aud_input = gr.Audio(
                        type="filepath",
                        label=TRANSLATIONS["en"]["record_audio"],
                        sources=["microphone", "upload"],
                        scale=1,
                    )



        # -------------------- Events --------------------
        def refresh_sessions_and_history(current_value: int = None):
            """Refresh session list and load history for selected session."""
            try:
                choices = session_manager.list_session_choices()
                if not choices:
                    # If no choices, get fresh list, ChatDatabase ensures at least one session exists
                    choices = session_manager.list_session_choices()
                
                if current_value and current_value in [choice[1] for choice in choices]:
                    selected_value = current_value
                else:
                    selected_value = choices[0][1] if choices else None
                
                if selected_value:
                    history = session_manager.get_history(selected_value, translator=orchestrator.translator)
                else:
                    history = []
                
                return gr.update(choices=choices, value=selected_value), history
            except Exception as e:
                print(f"Error in refresh_sessions_and_history: {e}")
                # Fallback to empty state
                return gr.update(choices=[], value=None), []

        def on_new_chat():
            """Handle new chat creation."""
            try:
                sid = session_manager.new_session()
                choices = session_manager.list_session_choices()
                history = session_manager.get_history(sid, translator=orchestrator.translator)
                return gr.update(choices=choices, value=sid), history
            except Exception as e:
                print(f"Error in on_new_chat: {e}")
                # Fallback to current state
                choices = session_manager.list_session_choices()
                current_value = choices[0][1] if choices else None
                history = session_manager.get_history(current_value, translator=orchestrator.translator) if current_value else []
                return gr.update(choices=choices, value=current_value), history

        def on_delete_chat(current_value: int):
            """Handle chat deletion."""
            try:
                _new_active = session_manager.delete_session(current_value)
                return refresh_sessions_and_history()
            except Exception as e:
                print(f"Error in on_delete_chat: {e}")
                # Fallback to current state
                return refresh_sessions_and_history()

        def on_select_chat(selected_value: int):
            """Handle chat selection."""
            return refresh_sessions_and_history(selected_value)

        def on_language_change(lang_choice):
            """Handle language change using translation dictionaries."""
            # Map our language choice to translation keys
            lang_map = {
                "english": "en",
                "urdu": "ur"
            }
            
            current_lang = lang_map.get(lang_choice, "en")
            translations = TRANSLATIONS[current_lang]
            
            # Update all components with new translations
            return (
                gr.update(choices=[(translations["english"], "english"), (translations["urdu"], "urdu")], value=lang_choice),
                gr.update(value=translations["new_chat"]),
                gr.update(label=translations["chats"]),
                gr.update(value=translations["delete_chat"]),
                gr.update(label=translations["speak_replies"]),
                gr.update(label=translations["tutor_speech"]),
                gr.update(label=translations["ai_tutor"]),
                gr.update(placeholder=translations["input_placeholder"]),
                gr.update(value=translations["send"]),
                gr.update(label=translations["upload_image"]),
                gr.update(label=translations["record_audio"]),
            )

        def user_handler(user_text, image_path, audio_path, selected_value, lang_choice, speak, history):
            """Handle user input (non-status)."""
            if lang_choice == "urdu":
                lang = 'ur'
            else:
                lang = 'en'
            
            # Ensure we're on the correct session
            session_manager.set_current_session_id(selected_value)
            
            return orchestrator.user_message(user_text, image_path, audio_path, selected_value, lang, history, session_manager)

        def bot_handler(selected_value, lang_choice, speak, history):
            """Handle bot response (non-status)."""
            if lang_choice == "urdu":
                lang = 'ur'
            else:
                lang = 'en'
            for updated_history, audio_file in orchestrator.llm_response(selected_value, lang, speak, history, session_manager):
                yield updated_history, audio_file

        # Bind buttons
        new_chat_btn.click(fn=on_new_chat, outputs=[chat_radio, chatbot])
        delete_btn.click(fn=on_delete_chat, inputs=[chat_radio], outputs=[chat_radio, chatbot])
        chat_radio.change(fn=on_select_chat, inputs=[chat_radio], outputs=[chat_radio, chatbot])

        lang_selector.change(
            fn=on_language_change,
            inputs=[lang_selector],
            outputs=[
                lang_selector,
                new_chat_btn,
                chat_radio,
                delete_btn,
                speak_toggle,
                tts_out,
                chatbot,
                txt_input,
                send_btn,
                img_input,
                aud_input,
            ]
        )
        # Send via button or enter - using chained events
        send_btn.click(
            fn=user_handler,
            inputs=[txt_input, img_input, aud_input, chat_radio, lang_selector, speak_toggle, chatbot],
            outputs=[txt_input, chatbot, img_input, aud_input, chat_radio],
            queue=False
        ).then(
            fn=bot_handler,
            inputs=[chat_radio, lang_selector, speak_toggle, chatbot],
            outputs=[chatbot, tts_out]
        )
        
        txt_input.submit(
            fn=user_handler,
            inputs=[txt_input, img_input, aud_input, chat_radio, lang_selector, speak_toggle, chatbot],
            outputs=[txt_input, chatbot, img_input, aud_input, chat_radio],
            queue=False
        ).then(
            fn=bot_handler,
            inputs=[chat_radio, lang_selector, speak_toggle, chatbot],
            outputs=[chatbot, tts_out]
        )

    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(i18n=i18n)
