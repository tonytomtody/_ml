import sys
import requests
import json
import os
from datetime import datetime
from threading import Thread
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QComboBox, QLabel, QScrollArea,
    QFrame, QFileDialog, QMessageBox, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer, QSize
from PyQt6.QtGui import QColor, QPalette, QCursor


# ====== AnythingLLM 設定 ======
API_KEY = "CZFHHBH-WMAMVHZ-HYKPJ7E-0Y25CJD"
BASE_URL = "http://localhost:3001/api/v1"

WORKSPACES = {
    "use CPU": "use-cpu",
    "use NPU": "30170325-6370-434f-838d-ff8a32aaacab",
}

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "accept": "application/json"
}

def online_check():
    try:
        online_status = requests.get("https://www.google.com", timeout=5)
        return True
    except:
        return False


class ResponseSignal(QObject):
    """Signal emitter for async API responses"""
    response_ready = pyqtSignal(str, list, float)  # reply, sources, response_time
    response_error = pyqtSignal(str)


def ask_anythingllm(workspace_slug: str, message: str, onlineStatus: bool):
    url = f"{BASE_URL}/workspace/{workspace_slug}/chat"

    if onlineStatus:
        message = "@agent " + message
    payload = {"message": message, "mode": "query"}

    try:
        resp = requests.post(url, headers=HEADERS, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        text_response = data.get("textResponse") or data.get("message") or str(data)
        sources = data.get("sources") or []
        return text_response, sources
    except requests.exceptions.ConnectionError:
        return "[Error] Cannot connect to AnythingLLM. Please ensure the server is running at http://localhost:3001\n\nSuggestion: Start AnythingLLM server and try again.", []
    except requests.exceptions.Timeout:
        return "[Error] Request timeout (2 minutes). The response took too long.\n\nSuggestion: Try a simpler question or check server performance.", []
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            return "[Error] Authentication failed. Invalid API key.\n\nSuggestion: Check your API_KEY in the configuration.", []
        elif e.response.status_code == 404:
            return "[Error] Workspace not found.\n\nSuggestion: Verify the workspace ID is correct.", []
        else:
            return f"[Error] HTTP Error {e.response.status_code}: {str(e)}\n\nSuggestion: Check server status and try again.", []
    except json.JSONDecodeError:
        return "[Error] Invalid response format from server.\n\nSuggestion: AnythingLLM may need restart.", []
    except Exception as e:
        error_msg = str(e)
        if "Workspace" in error_msg:
            return f"[Error] Workspace issue: {error_msg}\n\nSuggestion: Reload the workspace in AnythingLLM.", []
        else:
            return f"[Error] Unexpected error: {error_msg}\n\nSuggestion: Check logs and restart the application.", []


# ====== UI Component: Loading Indicator ======
class LoadingBubble(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.NoFrame)
        
        # Main outer layout (vertical)
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 5, 0, 5)
        outer_layout.setSpacing(0)

        # Create a horizontal layout for alignment
        h_layout = QHBoxLayout()
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(5)

        # Inner vertical layout for content
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(5, 5, 5, 5)
        content_layout.setSpacing(5)
        
        self.label = QLabel("⏳ Loading...")
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.label.setMinimumHeight(40)
        self.label.setStyleSheet(
            "padding: 10px; border-radius: 10px; font-size: 15px;"
            + "background-color: #2f2f2f; color: #888888;"
        )
        content_layout.addWidget(self.label)
        
        # Animate the dots
        self.timer = QTimer()
        self.timer.timeout.connect(self.animate)
        self.dots = 0
        self.timer.start(500)
        
        # Create a container widget for the content
        content_widget = QWidget()
        content_widget.setLayout(content_layout)

        # Add to horizontal layout (loading always left-aligned)
        h_layout.addWidget(content_widget)
        h_layout.addStretch()

        outer_layout.addLayout(h_layout)
        self.setLayout(outer_layout)
    
    def animate(self):
        self.dots = (self.dots + 1) % 4
        dots_text = "." * self.dots
        self.label.setText("⏳ Loading" + dots_text)
    
    def stop(self):
        self.timer.stop()


# ====== UI Component: Chat Bubble ======
class ChatBubble(QFrame):
    def __init__(self, text, is_user=False, isOnline=False, sources=None, timestamp=None, response_time=None):
        super().__init__()
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.sources = sources or []
        self.is_user = is_user
        self.timestamp = timestamp or datetime.now().strftime("%H:%M:%S")
        self.response_time = response_time

        if not is_user:
            background_color = "#2f2f2f"
        elif isOnline:
            background_color = "#1B5EAD"
        else:
            background_color = "#5EAD1B"

        # Main outer layout (vertical)
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 5, 0, 5)
        outer_layout.setSpacing(0)

        # Create a horizontal layout for alignment
        h_layout = QHBoxLayout()
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(5)

        # Inner vertical layout for message content
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(5, 5, 5, 5)
        content_layout.setSpacing(5)

        # Main message with timestamp and response time
        message_text = text
        if not is_user:
            time_info = f"[{self.timestamp}"
            if self.response_time is not None:
                time_info += f" ⏱ {self.response_time:.2f}s"
            time_info += "] "
            message_text = time_info + text
        
        label = QLabel(message_text)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        label.setMinimumHeight(40)
        label.setStyleSheet(
            "padding: 10px; border-radius: 10px; font-size: 15px;"
            + f"background-color: {background_color}; color: white;"
        )
        content_layout.addWidget(label)

        # Display sources with better formatting
        if self.sources and not is_user:
            self._create_sources_widget(content_layout)

        # Create a container widget for the content
        content_widget = QWidget()
        content_widget.setLayout(content_layout)

        # Add to horizontal layout with appropriate alignment
        if is_user:
            h_layout.addStretch()
            h_layout.addWidget(content_widget)
        else:
            h_layout.addWidget(content_widget)
            h_layout.addStretch()

        outer_layout.addLayout(h_layout)
        self.setLayout(outer_layout)
    
    def _create_sources_widget(self, layout):
        """Create an improved sources display widget"""
        sources_container = QFrame()
        sources_container.setFrameShape(QFrame.Shape.StyledPanel)
        sources_layout = QVBoxLayout()
        sources_layout.setContentsMargins(8, 8, 8, 8)
        sources_layout.setSpacing(3)
        
        # Title
        title_label = QLabel("📚 Sources & References:")
        title_label.setStyleSheet("font-weight: bold; color: #90EE90; font-size: 12px;")
        sources_layout.addWidget(title_label)
        
        # Parse sources
        for i, source in enumerate(self.sources, 1):
            if isinstance(source, dict):
                source_text = source.get('title', source.get('name', str(source)))
                if 'url' in source:
                    source_text += f" 🔗"
            else:
                source_text = str(source)
            
            source_label = QLabel(f"  {i}. {source_text}")
            source_label.setWordWrap(True)
            source_label.setStyleSheet(
                "padding: 5px; color: #D3D3D3; font-size: 11px;"
            )
            sources_layout.addWidget(source_label)
        
        sources_container.setStyleSheet(
            "background-color: #3f3f3f; border-left: 3px solid #90EE90; border-radius: 5px;"
        )
        sources_container.setLayout(sources_layout)
        layout.addWidget(sources_container)


# ====== Main Window ======
class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cooking Assistant - AnythingLLM")
        self.resize(400, 600)

        # Initialize conversation history
        self.conversation_history = []
        self.data_dir = "chat_history"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.apply_dark_theme()

        main_layout = QVBoxLayout()

        # ===== Top Bar with Controls =====
        top_bar = QHBoxLayout()
        
        top_bar.addWidget(QLabel("Mode:"))
        self.workspace_box = QComboBox()
        self.workspace_box.addItems(WORKSPACES.keys())
        self.workspace_box.setCurrentIndex(1)  # Default to use NPU
        top_bar.addWidget(self.workspace_box)
        
        top_bar.addSpacing(10)
        
        # Menu button for export
        self.menu_btn = QPushButton("⋮ Menu")
        self.menu_btn.setFixedWidth(80)
        menu = QMenu(self)
        menu.addAction("Export Chat (TXT)", self.export_chat_txt)
        menu.addAction("Export Chat (JSON)", self.export_chat_json)
        menu.addAction("Clear History", self.clear_history)
        self.menu_btn.setMenu(menu)
        top_bar.addWidget(self.menu_btn)
        
        top_bar.addStretch()
        
        # Status indicator
        self.status_label = QLabel("🟢 Online")
        self.status_label.setStyleSheet("color: #90EE90; font-weight: bold;")
        top_bar.addWidget(self.status_label)

        main_layout.addLayout(top_bar)

        # ===== Chat Area (Scrollable) =====
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; }")

        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout()
        self.chat_layout.setSpacing(5)
        self.chat_layout.setContentsMargins(5, 5, 5, 5)
        self.chat_layout.addStretch()
        self.chat_container.setLayout(self.chat_layout)

        self.scroll_area.setWidget(self.chat_container)
        main_layout.addWidget(self.scroll_area)

        # ===== Input Area =====
        bottom = QHBoxLayout()

        self.input_box = QTextEdit()
        self.input_box.setFixedHeight(80)
        self.input_box.setPlaceholderText("Ask me about recipes, cooking techniques, ingredients...")
        bottom.addWidget(self.input_box)

        button_layout = QVBoxLayout()
        self.send_btn = QPushButton("Send")
        self.send_btn.setFixedWidth(80)
        self.send_btn.clicked.connect(self.send_message)
        self.update_status()
        button_layout.addWidget(self.send_btn)
        button_layout.addStretch()
        
        bottom.addLayout(button_layout)

        main_layout.addLayout(bottom)

        self.setLayout(main_layout)
        
        # Signal for async responses
        self.response_signal = ResponseSignal()
        self.response_signal.response_ready.connect(self.on_response_ready)
        self.response_signal.response_error.connect(self.on_response_error)
        
        self.current_loading_bubble = None

    # ===== Dark Theme =====
    def apply_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#1e1e1e"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#2b2b2b"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.Button, QColor("#3c3c3c"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#ffffff"))
        self.setPalette(palette)

    # ===== Update Status Indicator =====
    def update_status(self):
        self.isOnline = online_check()
        if self.isOnline:
            self.status_label.setText("🟢 Online (Agent)")
            self.status_label.setStyleSheet("color: #90EE90; font-weight: bold;")
            self.send_btn.setStyleSheet("background-color: #1B5EAD; color: white; font-size: 14px; font-weight: bold;")
        else:
            self.status_label.setText("🟡 Offline (RAG)")
            self.status_label.setStyleSheet("color: #FFD700; font-weight: bold;")
            self.send_btn.setStyleSheet("background-color: #5EAD1B; color: white; font-size: 14px; font-weight: bold;")

    # ===== Add Message =====
    def add_message(self, text, is_user=False, sources=None, response_time=None):
        timestamp = datetime.now().strftime("%H:%M:%S")
        bubble = ChatBubble(text, is_user, self.isOnline, sources, timestamp, response_time)
        # Add message bubble with 0 stretch to keep messages compact
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble, 0)
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )
        
        # Save to history
        history_entry = {
            "timestamp": timestamp,
            "sender": "user" if is_user else "assistant",
            "message": text,
            "sources": sources or [],
            "response_time": response_time
        }
        self.conversation_history.append(history_entry)
        
    def send_message(self):
        user_text = self.input_box.toPlainText().strip()
        if not user_text:
            return

        self.update_status()
        
        # Add user message
        self.add_message(user_text, is_user=True)
        self.input_box.clear()
        
        # Show loading bubble
        self.current_loading_bubble = LoadingBubble()
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, self.current_loading_bubble)
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )
        
        # Disable send button during request
        self.send_btn.setEnabled(False)
        
        # Record start time for response tracking
        self.request_start_time = datetime.now()
        
        # Fetch response in background thread
        mode = self.workspace_box.currentText()
        slug = WORKSPACES[mode]
        
        thread = Thread(target=self._fetch_response, args=(slug, user_text, self.isOnline))
        thread.daemon = True
        thread.start()

    def _fetch_response(self, slug, message, is_online):
        """Fetch response from API in background"""
        reply, sources = ask_anythingllm(slug, message, is_online)
        # Calculate response time
        response_time = (datetime.now() - self.request_start_time).total_seconds()
        self.response_signal.response_ready.emit(reply, sources, response_time)

    def on_response_ready(self, reply, sources, response_time):
        """Handle API response"""
        # Remove loading bubble
        if self.current_loading_bubble:
            self.chat_layout.removeWidget(self.current_loading_bubble)
            self.current_loading_bubble.stop()
            self.current_loading_bubble.deleteLater()
            self.current_loading_bubble = None
        
        # Add response message with response time
        self.add_message(reply, is_user=False, sources=sources, response_time=response_time)
        
        # Enable send button
        self.send_btn.setEnabled(True)
        
        # Auto-scroll to bottom
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )

    def on_response_error(self, error_msg):
        """Handle API error"""
        # Remove loading bubble
        if self.current_loading_bubble:
            self.chat_layout.removeWidget(self.current_loading_bubble)
            self.current_loading_bubble.stop()
            self.current_loading_bubble.deleteLater()
            self.current_loading_bubble = None
        
        # Add error message
        self.add_message(error_msg, is_user=False)
        
        # Enable send button
        self.send_btn.setEnabled(True)

    # ===== Export Chat History (TXT) =====
    def export_chat_txt(self):
        """Export chat history as plain text"""
        if not self.conversation_history:
            QMessageBox.warning(self, "No History", "No messages to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Chat History", 
            os.path.join(self.data_dir, f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"),
            "Text Files (*.txt)"
        )
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write(f"Cooking Assistant Chat History\n")
                f.write(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                
                for entry in self.conversation_history:
                    sender = "You" if entry["sender"] == "user" else "Assistant"
                    f.write(f"[{entry['timestamp']}] {sender}:\n")
                    f.write(f"{entry['message']}\n")
                    
                    if entry['sources']:
                        f.write("\nSources:\n")
                        for i, source in enumerate(entry['sources'], 1):
                            if isinstance(source, dict):
                                source_text = source.get('title', source.get('name', str(source)))
                            else:
                                source_text = str(source)
                            f.write(f"  {i}. {source_text}\n")
                    f.write("\n" + "-" * 40 + "\n\n")
            
            QMessageBox.information(self, "Success", f"Chat history exported to:\n{file_path}")

    # ===== Export Chat History (JSON) =====
    def export_chat_json(self):
        """Export chat history as JSON"""
        if not self.conversation_history:
            QMessageBox.warning(self, "No History", "No messages to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Chat History",
            os.path.join(self.data_dir, f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"),
            "JSON Files (*.json)"
        )
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
            
            QMessageBox.information(self, "Success", f"Chat history exported to:\n{file_path}")

    # ===== Clear History =====
    def clear_history(self):
        """Clear conversation history"""
        reply = QMessageBox.question(
            self, "Clear History",
            "Are you sure you want to clear all chat history?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.conversation_history = []
            # Clear chat display
            while self.chat_layout.count() > 1:
                widget = self.chat_layout.takeAt(0)
                if widget:
                    widget.widget().deleteLater()
            QMessageBox.information(self, "Cleared", "Chat history cleared.")


# ===== Program Entry =====
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec())
