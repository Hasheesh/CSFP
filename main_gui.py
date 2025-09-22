"""
main_gui.py

This file creates the gradio interface for the ai tutor.
It uses the orchestrator.py file to handle the logic for the model calls and database management.


The parts of code not written by me are referenced from the following sources:
- gradio documentation to create the interface from https://www.gradio.app/docs/gradio/interface
- the css instructions from the gradio documentation 
  to implement the rtl styling from https://www.gradio.app/guides/custom-CSS-and-JS
- rtl sytling for urdu text code from https://medium.com/techradiant/quick-guideline-for-rtl-ui-2da60615b655
- implementation of ui language change from https://www.gradio.app/guides/internationalization

"""
import os
import logging
import warnings

# disable gpu usage for raspberry pi
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# disable warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Suppress various logging outputs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)



import gradio as gr
import gc
import os
from orchestrator import Orchestrator
from session_manager import SessionManager
from config_loader import get_translations, get_grade_options, get_subject_options, get_user_profile, save_user_profile

# Define the absolute path for the database file
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "chat_sessions.sqlite3")

# load translation dictionaries from config
TRANSLATIONS = get_translations()

# create an I18n instance with translations for English and Urdu
i18n = gr.I18n(**TRANSLATIONS)

def create_gradio_interface():
    """Create and configure the Gradio interface."""
    
    # load saved user profile
    user_profile = get_user_profile()
    saved_language = user_profile.get("language")
    saved_grade = user_profile.get("grade")
    saved_subject = user_profile.get("subject")
    saved_version = user_profile.get("version", "x86_64")

    # initialize the main orchestrator and session manager
    orchestrator = Orchestrator(version=saved_version) 
    session_manager = SessionManager(db_path=DB_PATH)
    
    css = """
    /*  RTL styling helper classes */ 
    .rtl {
        direction: rtl ;
        text-align: right ;
        unicode-bidi: embed ;
    }

    /* Style for Urdu text in chatbot using elem_id */
    #chatbot div[dir="rtl"] {
        direction: rtl;
        text-align: right;
        unicode-bidi: embed;
        font-family: 'Noto Nastaliq Urdu', 'Arial Unicode MS', sans-serif;
        font-size: 1.1em;
        line-height: 1.7;
    }
    """
    
    # create the gradio interface with chatbot and sidebar
    with gr.Blocks(title="AI Tutor", theme=gr.themes.Default(), css=css) as demo:
        gr.Markdown(TRANSLATIONS["en"]["title"])

        with gr.Row():
            # sidebar
            with gr.Column(scale=1, min_width=260):
                # create the version selector
                version_choices = ["x86_64", "arm"]
                version_selector = gr.Dropdown(
                    choices=version_choices,
                    value=saved_version,
                    label="Version: ",
                    interactive=True,
                )
                
                # create the grade selector (always English numerals for values and labels)
                grade_choices = [(opt["label"], opt["value"]) for opt in get_grade_options()]    
                grade_selector = gr.Dropdown(
                    choices=grade_choices,
                    value=saved_grade,  
                    label="Grade: ",
                    interactive=True,
                )

                # create the subject selector from saved language
                subject_choices = [(opt["label"], opt["value"]) for opt in get_subject_options(saved_language)]
                subject_selector = gr.Dropdown(
                    choices=subject_choices,
                    value=saved_subject,  
                    label="Subject: ",
                    interactive=True,
                )

                # map saved language to dropdown value
                lang_dropdown_value = "urdu" if saved_language == "ur" else "english"
                
                # create the language selector
                lang_selector = gr.Dropdown(
                    choices=[(TRANSLATIONS["en"]["english"], "english"), (TRANSLATIONS["en"]["urdu"], "urdu")],
                    value=lang_dropdown_value, 
                    label=TRANSLATIONS["en"]["language"],
                    interactive=True,
                )

                # create the speak toggle
                speak_toggle = gr.Checkbox(False, label=TRANSLATIONS["en"]["speak_replies"])

                # create the tts audio interface
                tts_out = gr.Audio(label=TRANSLATIONS["en"]["tutor_speech"], interactive=False)

                # create the new chat button
                new_chat_btn = gr.Button(TRANSLATIONS["en"]["new_chat"], variant="primary")

                # get the initial chat list from the database
                chat_choices = session_manager.list_session_choices()

                # chat_choices are lists [label, value]
                initial_value = chat_choices[0][1] 

                # create the chat list
                chat_radio = gr.Radio(
                    choices=chat_choices,
                    value=initial_value,
                    label=TRANSLATIONS["en"]["chats"],
                    interactive=True,
                )

                # create the delete button
                delete_btn = gr.Button(TRANSLATIONS["en"]["delete_chat"])

            # main area
            with gr.Column(scale=4):
                
                # get the initial session ID from the initial chat list
                initial_sid = initial_value  

                # create the chatbot interface
                # load initial chat with saved language preference
                initial_history = session_manager.get_history(
                    initial_sid, 
                    current_lang=saved_language)
                    
                chatbot = gr.Chatbot(
                    value=initial_history,
                    elem_id="chatbot",
                    label=TRANSLATIONS["en"]["ai_tutor"],
                    height=600,
                    show_copy_button=False,
                    type="messages",
                )

                with gr.Row():
                    with gr.Column(scale=5):
                        # create the text input interface
                        txt_input = gr.Textbox(
                            placeholder=TRANSLATIONS["en"]["input_placeholder"],
                            lines=1,
                            show_label=False,
                        )
                    with gr.Column(scale=1):
                        # create the send button
                        send_btn = gr.Button(TRANSLATIONS["en"]["send"], variant="primary")
                    with gr.Column(scale=1):
                        # create the stop button and make it non-interactive initially
                        stop_btn = gr.Button(TRANSLATIONS["en"]["stop"], interactive=False)


                with gr.Row():
                    # create the image input interface
                    img_input = gr.Image(
                        type="filepath",
                        label=TRANSLATIONS["en"]["upload_image"],
                        sources=["upload", "webcam", "clipboard"],
                        scale=1,
                        webcam_options=gr.WebcamOptions(mirror=False),
                    )
                    # create the audio input interface
                    aud_input = gr.Audio(
                        type="filepath",
                        label=TRANSLATIONS["en"]["record_audio"],
                        sources=["microphone", "upload"],
                        scale=1,
                    )


        # events
        def refresh_sessions_and_history(current_value=None, lang_choice=None):
            """Refresh session list and load history for selected session."""
            choices = session_manager.list_session_choices()
            if not choices:
                # if no choices, get fresh list, ChatDatabase ensures at least one session exists
                choices = session_manager.list_session_choices()

            # if current_value is None or not in the new choices, select the first available chat
            if current_value is None or not any(choice[1] == current_value for choice in choices):
                selected_value = choices[0][1]
            else:
                selected_value = current_value
            
            if selected_value:
                # determine current language
                current_lang = 'ur' if lang_choice == 'urdu' else 'en'
                history = session_manager.get_history(
                    selected_value, 
                    current_lang=current_lang)
            else:
                history = []
            
            return gr.update(choices=choices, value=selected_value), history


        def on_new_chat(lang_choice):
            """Handle new chat creation."""
            sid = session_manager.new_session()
            choices = session_manager.list_session_choices()
            current_lang = 'ur' if lang_choice == 'urdu' else 'en'
            history = session_manager.get_history(
                sid, 
                current_lang=current_lang,
            )
            gc.collect()
            return gr.update(choices=choices, value=sid), history
            
        def on_delete_chat(current_value: int, lang_choice: str):
            """Handle chat deletion."""
            # Delete the session and get the new active session ID
            new_session_id = session_manager.delete_session(current_value)
            gc.collect()
            # After deletion, refresh and auto-select the new session
            return refresh_sessions_and_history(current_value=new_session_id, lang_choice=lang_choice)


        def on_select_chat(selected_value: int, lang_choice: str):
            """Handle chat selection."""
            return refresh_sessions_and_history(selected_value, lang_choice)

        def on_language_change(lang_choice, current_chat_id):
            """Handle language change using translation dictionaries."""
            # map the language choice to translation keys
            if lang_choice == "urdu":
                current_lang = "ur"
            else:
                current_lang = "en"
            
            # save the language preference
            save_user_profile(language=current_lang)
            
            # reload the current chat with new language
            history = session_manager.get_history(
                current_chat_id,
                current_lang=current_lang)
                
            translations = TRANSLATIONS[current_lang]
            # update grade and subject choices
            grade_opts = [(opt["label"], opt["value"]) for opt in get_grade_options()]
            subject_opts = [(opt["label"], opt["value"]) for opt in get_subject_options(current_lang)]
            
            # update all UI components with new translations
            return (
                gr.update(choices=[(translations["english"], "english"), (translations["urdu"], "urdu")], value=lang_choice),
                gr.update(choices=grade_opts),
                gr.update(choices=subject_opts),
                gr.update(value=translations["new_chat"]),
                gr.update(label=translations["chats"]),
                gr.update(value=translations["delete_chat"]),
                gr.update(label=translations["speak_replies"]),
                gr.update(label=translations["tutor_speech"]),
                gr.update(label=translations["ai_tutor"]),
                gr.update(placeholder=translations["input_placeholder"]),
                gr.update(value=translations["send"]),
                gr.update(value=translations["stop"]),
                gr.update(label=translations["upload_image"]),
                gr.update(label=translations["record_audio"]),
                history  
            )
        
        def on_version_change(version_value):
            """Handle version selection change."""
            save_user_profile(version=version_value)
            orchestrator.switch_version(version_value)
            gr.Info(f"Version changed to {version_value}. Models reloaded.")

        def on_grade_change(grade_value):
            """Handle grade selection change."""
            save_user_profile(grade=grade_value)
            return grade_value
        
        def on_subject_change(subject_value):
            """Handle subject selection change."""
            save_user_profile(subject=subject_value)
            return subject_value

        def user_handler(user_text, image_path, audio_path, selected_value, lang_choice, history):
            """Handle user input, pass to orchestrator, and update button states."""
            if lang_choice == "urdu":
                lang = 'ur'
            else:
                lang = 'en'
            
            # ensure we're on the correct session
            session_manager.set_current_session_id(selected_value)
            
            # get updates from the orchestrator
            txt_out, hist_out, img_out, aud_out, radio_out = orchestrator.user_message(
                user_text, image_path, audio_path, selected_value, lang, history, session_manager
            )

            # disable send, enable stop
            return txt_out, hist_out, img_out, aud_out, radio_out, gr.update(interactive=False), gr.update(interactive=True)

        def bot_handler(selected_value, lang_choice, grade, subject, speak, history):
            """Handle bot response generation and updates button states during and after."""
            if lang_choice == "urdu":
                lang = 'ur'
            else:
                lang = 'en'

            # get the generator for the streaming response
            response_generator = orchestrator.llm_response(
                selected_value, lang, grade, subject, speak, history, session_manager
            )

            # yield updates as they come in
            final_history, final_audio_file = history, None
            for updated_history, audio_file in response_generator:
                final_history = updated_history
                final_audio_file = audio_file
                yield final_history, final_audio_file, gr.update(interactive=False), gr.update(interactive=True)
            
            gc.collect()
            # final update to re-enable send and disable stop
            yield final_history, final_audio_file, gr.update(interactive=True), gr.update(interactive=False)

        def stop_llm_stream():
            """Stop the LLM response stream. Does not manage UI state."""
            orchestrator.stop_llm_generation()

        # event handlers
        new_chat_btn.click(fn=on_new_chat, inputs=[lang_selector], outputs=[chat_radio, chatbot])
        delete_btn.click(fn=on_delete_chat, inputs=[chat_radio, lang_selector], outputs=[chat_radio, chatbot])
        chat_radio.change(fn=on_select_chat, inputs=[chat_radio, lang_selector], outputs=[chat_radio, chatbot])

        version_selector.change(
            fn=on_version_change,
            inputs=[version_selector],
            outputs=[]
        )

        lang_selector.change(
            fn=on_language_change,
            inputs=[lang_selector, chat_radio],
            outputs=[
                lang_selector,
                grade_selector,
                subject_selector,
                new_chat_btn,
                chat_radio,
                delete_btn,
                speak_toggle,
                tts_out,
                chatbot,
                txt_input,
                send_btn,
                stop_btn,
                img_input,
                aud_input,
                chatbot,
            ]
        )
        
        # save preferences when grade or subject changes
        grade_selector.change(
            fn=on_grade_change,
            inputs=[grade_selector],
            outputs=[grade_selector]
        )
        
        subject_selector.change(
            fn=on_subject_change,
            inputs=[subject_selector],
            outputs=[subject_selector]
        )
        
        # define IO lists for chat interactions
        user_inputs = [txt_input, img_input, aud_input, chat_radio, lang_selector, chatbot]
        user_outputs = [txt_input, chatbot, img_input, aud_input, chat_radio, send_btn, stop_btn]
        bot_inputs = [chat_radio, lang_selector, grade_selector, subject_selector, speak_toggle, chatbot]
        bot_outputs = [chatbot, tts_out, send_btn, stop_btn]

        # chain events for text input submission
        # user_handler runs to process input and update UI
        # bot_handler runs after to generate the response
        txt_input.submit(
            fn=user_handler,
            inputs=user_inputs,
            outputs=user_outputs,
            queue=False
        ).then(
            fn=bot_handler,
            inputs=bot_inputs,
            outputs=bot_outputs
        )

        # chain events for send button click
        # user_handler runs to process input and update UI
        # bot_handler runs after to generate the response
        send_btn.click(
            fn=user_handler,
            inputs=user_inputs,
            outputs=user_outputs,
            queue=False
        ).then(
            fn=bot_handler,
            inputs=bot_inputs,
            outputs=bot_outputs
        )
        
        # event for stop button
        # sets a flag to stop the generation in the backend
        # the bot_handler then completes its final yield to update the button states
        stop_btn.click(
            fn=stop_llm_stream,
            inputs=None,
            outputs=None,
        )

    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.queue(
        max_size=8,            # small queue to avoid out of memory errors
        status_update_rate=0.2 # smoother UI updates for streaming
    ).launch(
        # server_name="0.0.0.0",
        # server_port=7860,
        share=False,
        i18n=i18n
    )
