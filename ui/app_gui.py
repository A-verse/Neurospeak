"""
🎯 Advanced NeuroSpeak GUI with Enhanced Features

Features:
- Multi-tab interface for different functions
- Voice customization panel
- Phrase management system
- User profiles
- Accessibility features
- Real-time feedback
- System monitoring
- Emergency protocols
- Comprehensive logging
- Multi-language support
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
from tkinter.font import Font
import json
import os
import logging
import platform
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import threading
import queue
import pygame
from pathlib import Path
from dataclasses import dataclass
import uuid
import sys

# Import enhanced speech modules
from audio.text_to_speech import tts_controller, VoiceGender
from audio.voice_selector import voice_manager
from ai.emotion_mapper import get_voice_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neurospeak_gui.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = Path.home() / ".neurospeak"
USER_PROFILES_FILE = CONFIG_DIR / "user_profiles.json"
PHRASES_FILE = CONFIG_DIR / "custom_phrases.json"
MAX_RECENT_PHRASES = 10
EMERGENCY_PHRASE = "Emergency! I need immediate assistance!"

@dataclass
class UserProfile:
    id: str
    name: str
    preferred_voice: Optional[str] = None
    preferred_gender: Optional[VoiceGender] = None
    preferred_rate: Optional[int] = None
    preferred_volume: Optional[float] = None
    custom_phrases: List[str] = None
    accessibility_settings: Dict = None

class NeuroSpeakGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NeuroSpeak Pro - Communication Interface")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        self.root.config(bg="#f0f2f5")
        
        # Initialize systems
        self._setup_directories()
        self._load_resources()
        self._init_audio()
        
        # GUI state
        self.current_user = None
        self.recent_phrases = []
        self.emergency_mode = False
        self.voice_test_active = False
        self.message_queue = queue.Queue()
        
        # Build UI
        self._create_menu()
        self._create_notebook()
        self._create_status_bar()
        
        # Start message processor
        self._process_messages()
        
        # System checks
        self._perform_system_checks()
    
    def _setup_directories(self):
        """Ensure all required directories exist."""
        try:
            CONFIG_DIR.mkdir(exist_ok=True)
        except Exception as e:
            logger.error(f"Could not create config directory: {str(e)}")
            messagebox.showerror(
                "Initialization Error",
                f"Could not create configuration directory: {str(e)}"
            )
            sys.exit(1)
    
    def _load_resources(self):
        """Load fonts, icons, and other resources."""
        try:
            # Load fonts
            self.title_font = Font(family="Helvetica", size=18, weight="bold")
            self.button_font = Font(family="Helvetica", size=12)
            self.small_font = Font(family="Helvetica", size=10)
            
            # Load color scheme
            self.colors = {
                'primary': '#4a6fa5',
                'secondary': '#166088',
                'accent': '#4fc3f7',
                'background': '#f0f2f5',
                'text': '#333333',
                'warning': '#ff5252',
                'emergency': '#d32f2f'
            }
            
            # Load phrases and profiles
            self.phrases = self._load_phrases()
            self.user_profiles = self._load_user_profiles()
            
        except Exception as e:
            logger.error(f"Resource loading failed: {str(e)}")
            messagebox.showerror(
                "Resource Error",
                f"Could not load required resources: {str(e)}"
            )
            sys.exit(1)
    
    def _init_audio(self):
        """Initialize audio systems."""
        try:
            pygame.mixer.init()
            self.audio_ready = True
        except Exception as e:
            logger.error(f"Audio initialization failed: {str(e)}")
            self.audio_ready = False
            messagebox.showwarning(
                "Audio Warning",
                "Some audio features may not work properly"
            )
    
    def _load_phrases(self) -> Dict:
        """Load phrases from file or return defaults."""
        default_phrases = {
            'basic': [
                "Yes", "No", "Thank you", "Please", 
                "I need help", "I'm hungry", "I'm thirsty",
                "Good morning", "Good night", "I love you"
            ],
            'medical': [
                "I need medication", "I'm in pain",
                "Call my doctor", "I feel dizzy",
                "I need to use the bathroom"
            ],
            'emotional': [
                "I'm happy", "I'm sad", "I'm frustrated",
                "I'm excited", "I'm scared", "I'm tired"
            ]
        }
        
        if PHRASES_FILE.exists():
            try:
                with open(PHRASES_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Could not load phrases: {str(e)}")
                return default_phrases
        return default_phrases
    
    def _load_user_profiles(self) -> Dict:
        """Load user profiles from file."""
        if USER_PROFILES_FILE.exists():
            try:
                with open(USER_PROFILES_FILE, 'r') as f:
                    profiles = json.load(f)
                    return {k: UserProfile(**v) for k, v in profiles.items()}
            except Exception as e:
                logger.error(f"Could not load user profiles: {str(e)}")
        return {}
    
    def _save_user_profiles(self):
        """Save user profiles to file."""
        try:
            with open(USER_PROFILES_FILE, 'w') as f:
                profiles = {k: vars(v) for k, v in self.user_profiles.items()}
                json.dump(profiles, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save user profiles: {str(e)}")
    
    def _create_menu(self):
        """Create the main menu bar."""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New User Profile", command=self._create_user_profile)
        file_menu.add_command(label="Load Profile", command=self._load_user_profile)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="Voice Settings", command=self._show_voice_settings)
        settings_menu.add_command(label="Accessibility", command=self._show_accessibility_settings)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        help_menu.add_command(label="System Info", command=self._show_system_info)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def _create_notebook(self):
        """Create the tabbed interface."""
        self.notebook = ttk.Notebook(self.root)
        
        # Create tabs
        self.main_tab = ttk.Frame(self.notebook)
        self.phrases_tab = ttk.Frame(self.notebook)
        self.voice_tab = ttk.Frame(self.notebook)
        self.monitor_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.main_tab, text="Quick Speak")
        self.notebook.add(self.phrases_tab, text="Phrase Manager")
        self.notebook.add(self.voice_tab, text="Voice Studio")
        self.notebook.add(self.monitor_tab, text="System Monitor")
        
        self.notebook.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Build each tab
        self._build_main_tab()
        self._build_phrases_tab()
        self._build_voice_tab()
        self._build_monitor_tab()
    
    def _build_main_tab(self):
        """Build the main quick speak tab."""
        # Header
        header = ttk.Label(
            self.main_tab,
            text="Quick Speak",
            font=self.title_font,
            anchor='center'
        )
        header.pack(fill='x', pady=10)
        
        # Emergency button
        emergency_btn = tk.Button(
            self.main_tab,
            text="🚨 EMERGENCY",
            font=Font(family="Helvetica", size=14, weight="bold"),
            bg=self.colors['emergency'],
            fg='white',
            command=self._trigger_emergency,
            height=2,
            width=20
        )
        emergency_btn.pack(pady=10)
        
        # Emotion selector
        emotion_frame = ttk.LabelFrame(self.main_tab, text="Speech Tone")
        emotion_frame.pack(fill='x', padx=10, pady=5)
        
        self.emotion_var = tk.StringVar(value="neutral")
        
        emotions = [
            "neutral", "urgent", "joyful", "sad", 
            "grateful", "gentle", "cheerful", "authoritative"
        ]
        
        for i, emotion in enumerate(emotions):
            btn = tk.Radiobutton(
                emotion_frame,
                text=emotion.capitalize(),
                variable=self.emotion_var,
                value=emotion,
                font=self.button_font,
                command=self._update_voice_preview
            )
            btn.grid(row=i//4, column=i%4, sticky='w', padx=5, pady=2)
        
        # Quick phrase buttons
        phrase_frame = ttk.LabelFrame(self.main_tab, text="Quick Phrases")
        phrase_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.phrase_buttons = []
        for category, phrases in self.phrases.items():
            cat_frame = ttk.Frame(phrase_frame)
            cat_frame.pack(fill='x', padx=5, pady=5, anchor='w')
            
            ttk.Label(cat_frame, text=category.capitalize() + ":").pack(side='left')
            
            for phrase in phrases[:8]:  # Limit to 8 per category
                btn = tk.Button(
                    cat_frame,
                    text=phrase,
                    font=self.button_font,
                    command=lambda p=phrase: self.speak_phrase(p),
                    width=15
                )
                btn.pack(side='left', padx=2, pady=2)
                self.phrase_buttons.append(btn)
        
        # Custom phrase entry
        custom_frame = ttk.Frame(self.main_tab)
        custom_frame.pack(fill='x', padx=10, pady=10)
        
        self.custom_entry = ttk.Entry(
            custom_frame,
            font=self.button_font
        )
        self.custom_entry.pack(side='left', fill='x', expand=True, padx=5)
        
        speak_btn = tk.Button(
            custom_frame,
            text="Speak",
            font=self.button_font,
            command=self._speak_custom,
            width=10
        )
        speak_btn.pack(side='right', padx=5)
    
    def _build_phrases_tab(self):
        """Build the phrase management tab."""
        # Header
        ttk.Label(
            self.phrases_tab,
            text="Manage Phrases",
            font=self.title_font
        ).pack(pady=10)
        
        # Phrase categories
        cat_frame = ttk.Frame(self.phrases_tab)
        cat_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(cat_frame, text="Category:").pack(side='left')
        
        self.category_var = tk.StringVar(value="basic")
        category_menu = ttk.Combobox(
            cat_frame,
            textvariable=self.category_var,
            values=list(self.phrases.keys()),
            state="readonly"
        )
        category_menu.pack(side='left', padx=5)
        category_menu.bind('<<ComboboxSelected>>', self._update_phrase_list)
        
        # Phrase list
        list_frame = ttk.Frame(self.phrases_tab)
        list_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.phrase_list = tk.Listbox(
            list_frame,
            font=self.button_font,
            selectmode='single',
            height=10
        )
        self.phrase_list.pack(side='left', fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical')
        scrollbar.config(command=self.phrase_list.yview)
        scrollbar.pack(side='right', fill='y')
        self.phrase_list.config(yscrollcommand=scrollbar.set)
        
        # Phrase controls
        control_frame = ttk.Frame(self.phrases_tab)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        self.new_phrase_entry = ttk.Entry(control_frame)
        self.new_phrase_entry.pack(side='left', fill='x', expand=True, padx=5)
        
        add_btn = tk.Button(
            control_frame,
            text="Add",
            command=self._add_phrase
        )
        add_btn.pack(side='left', padx=2)
        
        remove_btn = tk.Button(
            control_frame,
            text="Remove",
            command=self._remove_phrase
        )
        remove_btn.pack(side='left', padx=2)
        
        speak_btn = tk.Button(
            control_frame,
            text="Speak",
            command=self._speak_selected_phrase
        )
        speak_btn.pack(side='right', padx=2)
        
        # Initialize list
        self._update_phrase_list()
    
    def _build_voice_tab(self):
        """Build the voice customization tab."""
        # Header
        ttk.Label(
            self.voice_tab,
            text="Voice Studio",
            font=self.title_font
        ).pack(pady=10)
        
        # Voice selection
        voice_frame = ttk.LabelFrame(self.voice_tab, text="Voice Settings")
        voice_frame.pack(fill='x', padx=10, pady=5)
        
        # Gender selection
        ttk.Label(voice_frame, text="Gender:").grid(row=0, column=0, sticky='e')
        
        self.gender_var = tk.StringVar()
        gender_menu = ttk.Combobox(
            voice_frame,
            textvariable=self.gender_var,
            values=["Male", "Female", "Neutral"],
            state="readonly"
        )
        gender_menu.grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        gender_menu.bind('<<ComboboxSelected>>', self._update_voice_options)
        
        # Voice selection
        ttk.Label(voice_frame, text="Voice:").grid(row=1, column=0, sticky='e')
        
        self.voice_var = tk.StringVar()
        self.voice_menu = ttk.Combobox(
            voice_frame,
            textvariable=self.voice_var,
            state="readonly"
        )
        self.voice_menu.grid(row=1, column=1, sticky='ew', padx=5, pady=2)
        
        # Rate control
        ttk.Label(voice_frame, text="Speed:").grid(row=2, column=0, sticky='e')
        
        self.rate_var = tk.IntVar(value=150)
        rate_scale = ttk.Scale(
            voice_frame,
            from_=50,
            to=300,
            variable=self.rate_var,
            command=lambda _: self._update_voice_preview()
        )
        rate_scale.grid(row=2, column=1, sticky='ew', padx=5, pady=2)
        
        # Volume control
        ttk.Label(voice_frame, text="Volume:").grid(row=3, column=0, sticky='e')
        
        self.volume_var = tk.DoubleVar(value=1.0)
        volume_scale = ttk.Scale(
            voice_frame,
            from_=0.0,
            to=1.0,
            variable=self.volume_var,
            command=lambda _: self._update_voice_preview()
        )
        volume_scale.grid(row=3, column=1, sticky='ew', padx=5, pady=2)
        
        # Test button
        test_frame = ttk.Frame(voice_frame)
        test_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        self.test_btn = tk.Button(
            test_frame,
            text="Test Voice",
            command=self._test_voice_settings
        )
        self.test_btn.pack(side='left', padx=5)
        
        # Voice preview text
        self.preview_text = ttk.Entry(
            test_frame,
            font=self.button_font
        )
        self.preview_text.insert(0, "Hello, this is a voice preview")
        self.preview_text.pack(side='left', fill='x', expand=True, padx=5)
        
        # Initialize voice options
        self._update_voice_options()
    
    def _build_monitor_tab(self):
        """Build the system monitor tab."""
        # Header
        ttk.Label(
            self.monitor_tab,
            text="System Monitor",
            font=self.title_font
        ).pack(pady=10)
        
        # System info
        info_frame = ttk.LabelFrame(self.monitor_tab, text="System Information")
        info_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.system_info = scrolledtext.ScrolledText(
            info_frame,
            font=self.small_font,
            wrap=tk.WORD,
            height=10
        )
        self.system_info.pack(fill='both', expand=True, padx=5, pady=5)
        self.system_info.insert(tk.END, "System information will be displayed here")
        self.system_info.config(state='disabled')
        
        # Refresh button
        refresh_btn = tk.Button(
            self.monitor_tab,
            text="Refresh",
            command=self._update_system_info
        )
        refresh_btn.pack(pady=5)
        
        # Initial update
        self._update_system_info()
    
    def _create_status_bar(self):
        """Create the status bar at bottom of window."""
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        
        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief='sunken',
            anchor='w'
        )
        status_bar.pack(side='bottom', fill='x')
    
    def _update_status(self, message: str):
        """Update the status bar message."""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def _process_messages(self):
        """Process messages from the queue."""
        try:
            while True:
                message = self.message_queue.get_nowait()
                self._update_status(message)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._process_messages)
    
    def _perform_system_checks(self):
        """Run initial system checks."""
        checks = [
            ("Audio System", self.audio_ready),
            ("TTS Engine", tts_controller._active),
            ("Voice Database", voice_manager._voices_loaded)
        ]
        
        for name, status in checks:
            if not status:
                self.message_queue.put(f"⚠️ Warning: {name} not fully initialized")
    
    def _update_phrase_list(self, event=None):
        """Update the phrase list with current category."""
        category = self.category_var.get()
        self.phrase_list.delete(0, tk.END)
        
        for phrase in self.phrases.get(category, []):
            self.phrase_list.insert(tk.END, phrase)
    
    def _add_phrase(self):
        """Add a new phrase to the current category."""
        phrase = self.new_phrase_entry.get().strip()
        category = self.category_var.get()
        
        if phrase and category:
            if phrase not in self.phrases[category]:
                self.phrases[category].append(phrase)
                self._update_phrase_list()
                self.new_phrase_entry.delete(0, tk.END)
                
                # Save to file
                try:
                    with open(PHRASES_FILE, 'w') as f:
                        json.dump(self.phrases, f, indent=2)
                except Exception as e:
                    messagebox.showerror("Error", f"Could not save phrases: {str(e)}")
    
    def _remove_phrase(self):
        """Remove the selected phrase."""
        selection = self.phrase_list.curselection()
        category = self.category_var.get()
        
        if selection and category:
            index = selection[0]
            phrase = self.phrases[category][index]
            
            if messagebox.askyesno("Confirm", f"Remove phrase: {phrase}?"):
                del self.phrases[category][index]
                self._update_phrase_list()
                
                # Save to file
                try:
                    with open(PHRASES_FILE, 'w') as f:
                        json.dump(self.phrases, f, indent=2)
                except Exception as e:
                    messagebox.showerror("Error", f"Could not save phrases: {str(e)}")
    
    def _speak_selected_phrase(self):
        """Speak the currently selected phrase."""
        selection = self.phrase_list.curselection()
        category = self.category_var.get()
        
        if selection and category:
            phrase = self.phrases[category][selection[0]]
            self.speak_phrase(phrase)
    
    def _speak_custom(self):
        """Speak the custom phrase from entry."""
        phrase = self.custom_entry.get().strip()
        if phrase:
            self.speak_phrase(phrase)
    
    def speak_phrase(self, phrase: str, emotion: Optional[str] = None):
        """Speak a phrase with optional emotion."""
        if not phrase:
            return
        
        emotion = emotion or self.emotion_var.get()
        
        try:
            # Get voice settings for emotion
            config = get_voice_settings(emotion)
            
            # Apply voice settings
            tts_controller.set_voice_settings(
                rate=config["rate"],
                volume=config["volume"]
            )
            
            # Speak the phrase
            tts_controller.speak(phrase)
            
            # Add to recent phrases
            self._add_recent_phrase(phrase)
            
            self.message_queue.put(f"Speaking: {phrase}")
        except Exception as e:
            logger.error(f"Error speaking phrase: {str(e)}")
            messagebox.showerror("Error", f"Could not speak phrase: {str(e)}")
    
    def _add_recent_phrase(self, phrase: str):
        """Add a phrase to recent phrases list."""
        if phrase in self.recent_phrases:
            self.recent_phrases.remove(phrase)
        
        self.recent_phrases.insert(0, phrase)
        if len(self.recent_phrases) > MAX_RECENT_PHRASES:
            self.recent_phrases = self.recent_phrases[:MAX_RECENT_PHRASES]
    
    def _trigger_emergency(self):
        """Handle emergency button press."""
        self.emergency_mode = True
        
        # Visual feedback
        self.root.config(bg=self.colors['emergency'])
        self.notebook.config(style='Emergency.TNotebook')
        
        # Speak emergency phrase
        self.speak_phrase(EMERGENCY_PHRASE, "urgent")
        
        # Flash window
        self._flash_emergency()
        
        # Reset after 10 seconds
        self.root.after(10000, self._reset_emergency)
    
    def _flash_emergency(self, state=True):
        """Flash the window during emergency."""
        if self.emergency_mode:
            if state:
                self.root.config(bg=self.colors['emergency'])
            else:
                self.root.config(bg=self.colors['background'])
            self.root.after(500, self._flash_emergency, not state)
    
    def _reset_emergency(self):
        """Reset from emergency mode."""
        self.emergency_mode = False
        self.root.config(bg=self.colors['background'])
        self.notebook.config(style='TNotebook')
    
    def _update_voice_options(self, event=None):
        """Update voice options based on selected gender."""
        gender = self.gender_var.get().lower() if self.gender_var.get() else None
        
        voices = voice_manager.list_voices()
        if gender:
            voices = [v for v in voices if v.gender.name.lower() == gender]
        
        self.voice_menu['values'] = [v.name for v in voices]
        if voices:
            self.voice_var.set(voices[0].name)
    
    def _test_voice_settings(self):
        """Test current voice settings."""
        if self.voice_test_active:
            return
        
        self.voice_test_active = True
        self.test_btn.config(state='disabled')
        
        try:
            # Apply selected voice
            voice_name = self.voice_var.get()
            voices = voice_manager.list_voices()
            selected_voice = next((v for v in voices if v.name == voice_name), None)
            
            if selected_voice:
                voice_manager.select_voice(selected_voice.system_id, preview=False)
            
            # Apply rate and volume
            tts_controller.set_voice_settings(
                rate=self.rate_var.get(),
                volume=self.volume_var.get()
            )
            
            # Speak preview text
            text = self.preview_text.get()
            if text:
                tts_controller.speak(text)
        
        except Exception as e:
            logger.error(f"Voice test failed: {str(e)}")
            messagebox.showerror("Error", f"Voice test failed: {str(e)}")
        finally:
            self.voice_test_active = False
            self.test_btn.config(state='normal')
    
    def _update_voice_preview(self, event=None):
        """Update voice preview with current settings."""
        # This could show a visual representation of the voice settings
        pass
    
    def _update_system_info(self):
        """Update system information display."""
        info = [
            f"System: {platform.system()} {platform.release()}",
            f"Python: {platform.python_version()}",
            f"TTS Engine: {'Active' if tts_controller._active else 'Inactive'}",
            f"Voices Loaded: {len(voice_manager._profiles)}",
            f"User Profiles: {len(self.user_profiles)}",
            f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        
        self.system_info.config(state='normal')
        self.system_info.delete(1.0, tk.END)
        self.system_info.insert(tk.END, "\n".join(info))
        self.system_info.config(state='disabled')
    
    def _create_user_profile(self):
        """Create a new user profile."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Create User Profile")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Profile Name:").grid(row=0, column=0, padx=5, pady=5)
        name_entry = ttk.Entry(dialog)
        name_entry.grid(row=0, column=1, padx=5, pady=5)
        
        def save_profile():
            name = name_entry.get().strip()
            if name:
                profile_id = str(uuid.uuid4())
                self.user_profiles[profile_id] = UserProfile(
                    id=profile_id,
                    name=name
                )
                self._save_user_profiles()
                dialog.destroy()
                messagebox.showinfo("Success", "Profile created successfully")
        
        save_btn = ttk.Button(dialog, text="Save", command=save_profile)
        save_btn.grid(row=1, column=0, columnspan=2, pady=5)
    
    def _load_user_profile(self):
        """Load an existing user profile."""
        if not self.user_profiles:
            messagebox.showinfo("Info", "No user profiles available")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Load User Profile")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Select Profile:").grid(row=0, column=0, padx=5, pady=5)
        
        profile_var = tk.StringVar()
        profile_menu = ttk.Combobox(
            dialog,
            textvariable=profile_var,
            values=[p.name for p in self.user_profiles.values()],
            state="readonly"
        )
        profile_menu.grid(row=0, column=1, padx=5, pady=5)
        
        def load_profile():
            selected_name = profile_var.get()
            if selected_name:
                profile = next((p for p in self.user_profiles.values() if p.name == selected_name), None)
                if profile:
                    self.current_user = profile
                    self._apply_user_profile(profile)
                    dialog.destroy()
                    messagebox.showinfo("Success", f"Profile '{profile.name}' loaded")
        
        load_btn = ttk.Button(dialog, text="Load", command=load_profile)
        load_btn.grid(row=1, column=0, columnspan=2, pady=5)
    
    def _apply_user_profile(self, profile: UserProfile):
        """Apply settings from a user profile."""
        if profile.preferred_voice:
            voice_manager.select_voice(profile.preferred_voice)
        
        if profile.preferred_rate or profile.preferred_volume:
            tts_controller.set_voice_settings(
                rate=profile.preferred_rate,
                volume=profile.preferred_volume
            )
        
        self.message_queue.put(f"Profile loaded: {profile.name}")
    
    def _show_voice_settings(self):
        """Show advanced voice settings dialog."""
        messagebox.showinfo("Info", "Voice settings will be available in a future version")
    
    def _show_accessibility_settings(self):
        """Show accessibility settings dialog."""
        messagebox.showinfo("Info", "Accessibility settings will be available in a future version")
    
    def _show_about(self):
        """Show about dialog."""
        about_text = (
            "NeuroSpeak Pro\n"
            "Version 2.0\n\n"
            "Advanced communication interface\n"
            "for speech-impaired users\n\n"
            "© 2023 NeuroTech Solutions"
        )
        messagebox.showinfo("About NeuroSpeak", about_text)
    
    def _show_system_info(self):
        """Show detailed system information."""
        info = [
            f"Operating System: {platform.platform()}",
            f"Python Version: {sys.version}",
            f"TTS Engine: {tts_controller.get_stats()}",
            f"Voice Manager: {voice_manager.get_stats()}",
            f"GUI Version: 2.0",
            f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        
        messagebox.showinfo("System Information", "\n".join(info))

def main():
    """Main application entry point."""
    try:
        root = tk.Tk()
        
        # Create custom styles
        style = ttk.Style()
        style.configure('Emergency.TNotebook', background='#ffcccc')
        
        app = NeuroSpeakGUI(root)
        root.mainloop()
    except Exception as e:
        logger.critical(f"Application failed: {str(e)}")
        messagebox.showerror("Fatal Error", f"The application encountered a fatal error:\n{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()