from __future__ import annotations

import random
import time
import tkinter as tk
from datetime import datetime
from tkinter import ttk

from .config import APP_NAME, TASK_TIMING


class AssessmentApp:
    def __init__(self, participant_id: str, questionnaire: dict | None = None, baseline_session: bool = False) -> None:
        self.participant_id = participant_id
        self.questionnaire = questionnaire or {}
        self.baseline_session = baseline_session
        self.started_at = datetime.now()
        self.session: dict | None = None
        self.pending_jobs: list[str] = []
        self.active_key_handler = None
        self.results = {
            "reaction_time": {"trials": []},
            "task_switching": {"trials": []},
            "working_memory": {"trials": []},
            "stroop": {"trials": []},
        }

        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} - {participant_id}")
        self.root.geometry("980x680")
        self.root.configure(bg="#101820")
        self.root.bind("<KeyPress>", self._dispatch_keypress)
        self.root.protocol("WM_DELETE_WINDOW", self._abort)

        self.frame = tk.Frame(self.root, bg="#101820", padx=36, pady=32)
        self.frame.pack(fill="both", expand=True)

    def run(self) -> dict | None:
        self._show_intro()
        self.root.mainloop()
        return self.session

    def _abort(self) -> None:
        self.session = None
        self._cancel_jobs()
        self.root.quit()
        self.root.destroy()

    def _dispatch_keypress(self, event: tk.Event) -> None:
        if isinstance(event.widget, tk.Entry):
            return
        if self.active_key_handler is not None:
            self.active_key_handler(event)

    def _cancel_jobs(self) -> None:
        for job_id in self.pending_jobs:
            try:
                self.root.after_cancel(job_id)
            except tk.TclError:
                pass
        self.pending_jobs.clear()

    def _schedule(self, delay_ms: int, callback) -> str:
        job_ref: dict[str, str] = {}

        def wrapped() -> None:
            job_id = job_ref.get("id")
            if job_id in self.pending_jobs:
                self.pending_jobs.remove(job_id)
            callback()

        job_id = self.root.after(delay_ms, wrapped)
        job_ref["id"] = job_id
        self.pending_jobs.append(job_id)
        return job_id

    def _clear(self) -> None:
        self._cancel_jobs()
        self.active_key_handler = None
        for child in self.frame.winfo_children():
            child.destroy()

    def _seconds_text(self, milliseconds: int) -> str:
        seconds = milliseconds / 1000.0
        return f"{seconds:.1f} seconds"

    def _format_response_time(self, response_ms: float) -> str:
        return f"{response_ms / 1000.0:.2f} seconds"

    def _add_title(self, text: str) -> tk.Label:
        label = tk.Label(
            self.frame,
            text=text,
            bg="#101820",
            fg="#F2F7F2",
            font=("Segoe UI Semibold", 28),
            wraplength=820,
            justify="center",
        )
        label.pack(pady=(0, 18))
        return label

    def _add_body(self, text: str, font_size: int = 14, fg: str = "#D8E4D6") -> tk.Label:
        label = tk.Label(
            self.frame,
            text=text,
            bg="#101820",
            fg=fg,
            font=("Segoe UI", font_size),
            wraplength=860,
            justify="center",
        )
        label.pack(pady=(0, 12))
        return label

    def _show_message(self, title: str, body: str, button_text: str, on_continue) -> None:
        self._clear()
        self._add_title(title)
        self._add_body(body)
        button = ttk.Button(self.frame, text=button_text, command=on_continue)
        button.pack(pady=24)

    def _show_intro(self) -> None:
        baseline_text = "This session is marked as a baseline calibration session." if self.baseline_session else "This session will be stored as a regular assessment."
        self._show_message(
            title="Behavioral Cognitive Load Assessment",
            body=(
                f"Participant: {self.participant_id}\n\n"
                "You will complete four tasks measuring speed, flexibility, memory, and selective attention.\n"
                "Use the keyboard exactly as instructed on each screen.\n\n"
                f"{baseline_text}"
            ),
            button_text="Start",
            on_continue=self._show_reaction_instructions,
        )

    def _show_reaction_instructions(self) -> None:
        response_window = self._seconds_text(TASK_TIMING["reaction_time"]["response_window_ms"])
        self._show_message(
            title="Task 1: Reaction Time",
            body=(
                "Press SPACE only when the target turns green.\n"
                "If you press early, it counts as a false start.\n"
                f"Once the target turns green, you have up to {response_window} to respond.\n"
                "There are 8 trials."
            ),
            button_text="Begin Reaction Task",
            on_continue=self._start_reaction_task,
        )

    def _start_reaction_task(self) -> None:
        self._clear()
        self.reaction_response_window_ms = TASK_TIMING["reaction_time"]["response_window_ms"]
        self.reaction_feedback_delay_ms = TASK_TIMING["reaction_time"]["feedback_ms"]
        self.reaction_trial_token = 0
        self._add_title("Reaction Time")
        self.reaction_progress = tk.Label(self.frame, text="", bg="#101820", fg="#78C6A3", font=("Segoe UI", 13))
        self.reaction_progress.pack(pady=(0, 8))
        self.reaction_stimulus = tk.Label(self.frame, text="Wait for green", bg="#101820", fg="#D8E4D6", font=("Segoe UI Semibold", 34))
        self.reaction_stimulus.pack(pady=60)
        self.reaction_feedback = self._add_body(
            f"Press SPACE when the stimulus changes. You have up to {self._seconds_text(self.reaction_response_window_ms)}.",
            font_size=16,
        )
        self.reaction_trial_index = 0
        self.reaction_phase = "idle"
        self.reaction_signal_ts = 0.0
        self.active_key_handler = self._handle_reaction_key
        self._begin_reaction_trial()

    def _begin_reaction_trial(self) -> None:
        if self.reaction_trial_index >= 8:
            self._show_switching_instructions()
            return

        self.reaction_trial_token += 1
        self.reaction_phase = "waiting"
        self.reaction_progress.config(text=f"Trial {self.reaction_trial_index + 1} of 8")
        self.reaction_stimulus.config(text="Wait for green", fg="#D8E4D6")
        self.reaction_feedback.config(text="Stay ready. Press only after the green cue.")
        delay_ms = random.randint(1400, 3000)
        trial_token = self.reaction_trial_token
        self._schedule(delay_ms, lambda token=trial_token: self._show_reaction_signal(token))

    def _show_reaction_signal(self, trial_token: int) -> None:
        if trial_token != self.reaction_trial_token or self.reaction_phase != "waiting":
            return
        self.reaction_phase = "go"
        self.reaction_signal_ts = time.perf_counter()
        self.reaction_stimulus.config(text="PRESS SPACE NOW", fg="#59D185")
        self.reaction_feedback.config(
            text=f"Respond within {self._seconds_text(self.reaction_response_window_ms)}. Faster responses score better."
        )
        self._schedule(
            self.reaction_response_window_ms,
            lambda token=trial_token: self._reaction_timeout(token),
        )

    def _handle_reaction_key(self, event: tk.Event) -> None:
        if event.keysym.lower() != "space":
            return
        if self.reaction_phase == "waiting":
            self.results["reaction_time"]["trials"].append(
                {
                    "false_start": True,
                    "missed": False,
                    "response_ms": None,
                    "response_seconds": None,
                    "response_window_ms": self.reaction_response_window_ms,
                    "correct": False,
                }
            )
            self.reaction_phase = "feedback"
            self.reaction_feedback.config(text="False start recorded.", fg="#F66B6B")
            self.reaction_stimulus.config(text="Too early", fg="#F66B6B")
            self.reaction_trial_index += 1
            self._schedule(self.reaction_feedback_delay_ms, self._begin_reaction_trial)
        elif self.reaction_phase == "go":
            rt_ms = round((time.perf_counter() - self.reaction_signal_ts) * 1000.0, 2)
            self.results["reaction_time"]["trials"].append(
                {
                    "false_start": False,
                    "missed": False,
                    "response_ms": rt_ms,
                    "response_seconds": round(rt_ms / 1000.0, 4),
                    "response_window_ms": self.reaction_response_window_ms,
                    "correct": True,
                }
            )
            self.reaction_phase = "feedback"
            self.reaction_feedback.config(
                text=f"Recorded in {self._format_response_time(rt_ms)}.",
                fg="#78C6A3",
            )
            self.reaction_stimulus.config(text="Captured", fg="#78C6A3")
            self.reaction_trial_index += 1
            self._schedule(self.reaction_feedback_delay_ms, self._begin_reaction_trial)

    def _reaction_timeout(self, trial_token: int) -> None:
        if trial_token != self.reaction_trial_token or self.reaction_phase != "go":
            return
        self.results["reaction_time"]["trials"].append(
            {
                "false_start": False,
                "missed": True,
                "response_ms": None,
                "response_seconds": None,
                "response_window_ms": self.reaction_response_window_ms,
                "correct": False,
            }
        )
        self.reaction_phase = "feedback"
        self.reaction_feedback.config(
            text=f"No response within {self._seconds_text(self.reaction_response_window_ms)}.",
            fg="#F2B35B",
        )
        self.reaction_stimulus.config(text="Too slow", fg="#F2B35B")
        self.reaction_trial_index += 1
        self._schedule(self.reaction_feedback_delay_ms, self._begin_reaction_trial)

    def _show_switching_instructions(self) -> None:
        response_window = self._seconds_text(TASK_TIMING["task_switching"]["response_window_ms"])
        self._show_message(
            title="Task 2: Task Switching",
            body=(
                "You will classify either a number or a letter.\n"
                "If the cue says NUMBER: F = odd, J = even.\n"
                "If the cue says LETTER: F = vowel, J = consonant.\n"
                f"The cue changes across trials, so watch it closely.\n"
                f"Each stimulus stays open for up to {response_window}."
            ),
            button_text="Begin Switching Task",
            on_continue=self._start_switching_task,
        )

    def _build_switching_trials(self) -> list[dict]:
        trials = []
        previous_rule = random.choice(["number", "letter"])
        vowels = "AEIOU"
        consonants = "BCDFGHJKLMNPQRSTVWXYZ"
        for index in range(26):
            if index == 0:
                rule = previous_rule
                is_switch = False
            else:
                is_switch = random.random() < 0.5
                rule = "letter" if previous_rule == "number" and is_switch else "number" if is_switch else previous_rule
            previous_rule = rule
            letter = random.choice(vowels if random.random() < 0.4 else consonants)
            digit = random.choice("123456789")
            if rule == "number":
                correct_key = "f" if int(digit) % 2 == 1 else "j"
            else:
                correct_key = "f" if letter in vowels else "j"
            trials.append(
                {
                    "rule": rule,
                    "letter": letter,
                    "digit": digit,
                    "stimulus": f"{letter} {digit}",
                    "correct_key": correct_key,
                    "is_switch": is_switch,
                }
            )
        return trials

    def _start_switching_task(self) -> None:
        self._clear()
        self.switch_response_window_ms = TASK_TIMING["task_switching"]["response_window_ms"]
        self.switch_feedback_delay_ms = TASK_TIMING["task_switching"]["feedback_ms"]
        self.switch_trial_token = 0
        self._add_title("Task Switching")
        self.switch_progress = tk.Label(self.frame, text="", bg="#101820", fg="#78C6A3", font=("Segoe UI", 13))
        self.switch_progress.pack(pady=(0, 8))
        self.switch_cue = tk.Label(self.frame, text="", bg="#101820", fg="#8CC5FF", font=("Segoe UI Black", 28))
        self.switch_cue.pack(pady=(18, 10))
        self.switch_stimulus = tk.Label(self.frame, text="", bg="#101820", fg="#F2F7F2", font=("Segoe UI Semibold", 42))
        self.switch_stimulus.pack(pady=18)
        self.switch_mapping = self._add_body("NUMBER: F=Odd, J=Even    LETTER: F=Vowel, J=Consonant", font_size=15)
        self.switch_feedback = self._add_body(
            f"Respond with F or J. You have up to {self._seconds_text(self.switch_response_window_ms)}.",
            font_size=16,
        )
        self.switch_trials = self._build_switching_trials()
        self.switch_index = 0
        self.switch_phase = "idle"
        self.switch_start_ts = 0.0
        self.active_key_handler = self._handle_switch_key
        self._present_switch_trial()

    def _present_switch_trial(self) -> None:
        if self.switch_index >= len(self.switch_trials):
            self._show_memory_instructions()
            return

        self.switch_trial_token += 1
        trial = self.switch_trials[self.switch_index]
        self.switch_phase = "live"
        self.switch_progress.config(text=f"Trial {self.switch_index + 1} of {len(self.switch_trials)}")
        self.switch_cue.config(
            text=f"{trial['rule'].upper()} RULE",
            fg="#69B6FF" if trial["rule"] == "number" else "#FFAE57",
        )
        self.switch_stimulus.config(text=trial["stimulus"])
        self.switch_feedback.config(
            text=f"F or J - up to {self._seconds_text(self.switch_response_window_ms)}",
            fg="#D8E4D6",
        )
        self.switch_start_ts = time.perf_counter()
        trial_token = self.switch_trial_token
        self._schedule(
            self.switch_response_window_ms,
            lambda token=trial_token: self._switch_timeout(token),
        )

    def _record_switch_trial(self, pressed_key: str | None, missed: bool) -> None:
        trial = dict(self.switch_trials[self.switch_index])
        response_ms = None if missed else round((time.perf_counter() - self.switch_start_ts) * 1000.0, 2)
        trial.update(
            {
                "pressed_key": pressed_key,
                "response_ms": response_ms,
                "response_seconds": round(response_ms / 1000.0, 4) if response_ms is not None else None,
                "response_window_ms": self.switch_response_window_ms,
                "missed": missed,
                "correct": (pressed_key == trial["correct_key"]) and not missed,
            }
        )
        self.results["task_switching"]["trials"].append(trial)
        self.switch_index += 1
        self.switch_phase = "feedback"

    def _handle_switch_key(self, event: tk.Event) -> None:
        pressed_key = event.keysym.lower()
        if self.switch_phase != "live" or pressed_key not in {"f", "j"}:
            return

        self._record_switch_trial(pressed_key, missed=False)
        last_trial = self.results["task_switching"]["trials"][-1]
        if last_trial["correct"]:
            self.switch_feedback.config(
                text=f"Correct - responded in {self._format_response_time(last_trial['response_ms'])}.",
                fg="#78C6A3",
            )
        else:
            self.switch_feedback.config(
                text=f"Incorrect - responded in {self._format_response_time(last_trial['response_ms'])}.",
                fg="#F66B6B",
            )
        self._schedule(self.switch_feedback_delay_ms, self._present_switch_trial)

    def _switch_timeout(self, trial_token: int) -> None:
        if trial_token != self.switch_trial_token or self.switch_phase != "live":
            return
        self._record_switch_trial(pressed_key=None, missed=True)
        self.switch_feedback.config(
            text=f"No response within {self._seconds_text(self.switch_response_window_ms)}.",
            fg="#F2B35B",
        )
        self._schedule(self.switch_feedback_delay_ms, self._present_switch_trial)

    def _show_memory_instructions(self) -> None:
        recall_window = self._seconds_text(TASK_TIMING["working_memory"]["recall_window_ms"])
        self._show_message(
            title="Task 3: Working Memory",
            body=(
                "Watch the digit sequence carefully.\n"
                "After it disappears, type the sequence exactly and press Enter.\n"
                "Sequence length gradually increases from 3 to 7 digits.\n"
                f"You will have up to {recall_window} to type each answer."
            ),
            button_text="Begin Memory Task",
            on_continue=self._start_memory_task,
        )

    def _start_memory_task(self) -> None:
        self._clear()
        self.memory_digit_display_ms = TASK_TIMING["working_memory"]["digit_display_ms"]
        self.memory_digit_gap_ms = TASK_TIMING["working_memory"]["digit_gap_ms"]
        self.memory_recall_window_ms = TASK_TIMING["working_memory"]["recall_window_ms"]
        self.memory_feedback_delay_ms = TASK_TIMING["working_memory"]["feedback_ms"]
        self.memory_recall_token = 0
        self._add_title("Working Memory")
        self.memory_progress = tk.Label(self.frame, text="", bg="#101820", fg="#78C6A3", font=("Segoe UI", 13))
        self.memory_progress.pack(pady=(0, 8))
        self.memory_status = self._add_body("Memorize the sequence.", font_size=16)
        self.memory_digit = tk.Label(self.frame, text="", bg="#101820", fg="#F2F7F2", font=("Consolas", 54))
        self.memory_digit.pack(pady=48)
        self.memory_entry = ttk.Entry(self.frame, font=("Consolas", 20), justify="center")
        self.memory_entry.pack(pady=10, ipadx=6, ipady=6)
        self.memory_entry.bind("<Return>", self._submit_memory_response)
        self.memory_submit = ttk.Button(self.frame, text="Submit", command=self._submit_memory_response)
        self.memory_submit.pack(pady=(4, 16))
        self.memory_trials = [3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
        self.memory_index = 0
        self.memory_phase = "idle"
        self.memory_response_started = 0.0
        self.current_sequence = ""
        self.memory_entry.configure(state="disabled")
        self.memory_submit.configure(state="disabled")
        self._begin_memory_trial()

    def _begin_memory_trial(self) -> None:
        if self.memory_index >= len(self.memory_trials):
            self._show_stroop_instructions()
            return
        length = self.memory_trials[self.memory_index]
        self.current_sequence = "".join(random.choice("123456789") for _ in range(length))
        self.memory_phase = "display"
        self.memory_progress.config(text=f"Trial {self.memory_index + 1} of {len(self.memory_trials)}")
        self.memory_status.config(text=f"Sequence length: {length}", fg="#D8E4D6")
        self.memory_digit.config(text="")
        self.memory_entry.configure(state="disabled")
        self.memory_submit.configure(state="disabled")
        self.memory_entry.delete(0, tk.END)
        self._schedule(500, lambda: self._show_memory_digit(0))

    def _show_memory_digit(self, index: int) -> None:
        if index >= len(self.current_sequence):
            self.memory_digit.config(text="")
            self._schedule(300, self._enable_memory_entry)
            return
        self.memory_digit.config(text=self.current_sequence[index])
        self._schedule(self.memory_digit_display_ms, lambda idx=index: self._hide_memory_digit(idx))

    def _hide_memory_digit(self, index: int) -> None:
        self.memory_digit.config(text="")
        self._schedule(self.memory_digit_gap_ms, lambda idx=index + 1: self._show_memory_digit(idx))

    def _enable_memory_entry(self) -> None:
        self.memory_phase = "recall"
        self.memory_recall_token += 1
        self.memory_status.config(
            text=f"Type the sequence and press Enter. You have up to {self._seconds_text(self.memory_recall_window_ms)}.",
            fg="#D8E4D6",
        )
        self.memory_entry.configure(state="normal")
        self.memory_submit.configure(state="normal")
        self.memory_entry.focus_set()
        self.memory_response_started = time.perf_counter()
        recall_token = self.memory_recall_token
        self._schedule(
            self.memory_recall_window_ms,
            lambda token=recall_token: self._submit_memory_response(timeout=True, recall_token=token),
        )

    def _submit_memory_response(self, event=None, timeout: bool = False, recall_token: int | None = None) -> None:
        if recall_token is not None and recall_token != self.memory_recall_token:
            return
        if self.memory_phase != "recall":
            return
        response = self.memory_entry.get().strip()
        response_ms = round((time.perf_counter() - self.memory_response_started) * 1000.0, 2)
        position_matches = sum(
            1 for expected_digit, typed_digit in zip(self.current_sequence, response) if expected_digit == typed_digit
        )
        position_accuracy = position_matches / len(self.current_sequence)
        full_correct = response == self.current_sequence
        self.results["working_memory"]["trials"].append(
            {
                "sequence_length": len(self.current_sequence),
                "target_sequence": self.current_sequence,
                "response": response,
                "position_accuracy": round(position_accuracy, 4),
                "full_correct": full_correct,
                "response_ms": response_ms,
                "response_seconds": round(response_ms / 1000.0, 4),
                "response_window_ms": self.memory_recall_window_ms,
                "correct": full_correct,
                "missed": timeout and not response,
            }
        )
        self.memory_phase = "feedback"
        self.memory_entry.delete(0, tk.END)
        self.memory_entry.configure(state="disabled")
        self.memory_submit.configure(state="disabled")
        if full_correct:
            self.memory_status.config(
                text=f"Correct - responded in {self._format_response_time(response_ms)}.",
                fg="#78C6A3",
            )
        elif timeout and not response:
            self.memory_status.config(
                text=f"No response within {self._seconds_text(self.memory_recall_window_ms)}.",
                fg="#F2B35B",
            )
        else:
            self.memory_status.config(
                text=(
                    f"Expected {self.current_sequence} - responded in "
                    f"{self._format_response_time(response_ms)}."
                ),
                fg="#F2B35B",
            )
        self.memory_index += 1
        self._schedule(self.memory_feedback_delay_ms, self._begin_memory_trial)

    def _show_stroop_instructions(self) -> None:
        response_window = self._seconds_text(TASK_TIMING["stroop"]["response_window_ms"])
        self._show_message(
            title="Task 4: Selective Attention",
            body=(
                "Respond to the font color, not the word meaning.\n"
                "R = red, G = green, B = blue, Y = yellow.\n"
                f"Ignore what the word says. Follow the ink color only.\n"
                f"Each color stays open for up to {response_window}."
            ),
            button_text="Begin Stroop Task",
            on_continue=self._start_stroop_task,
        )

    def _build_stroop_trials(self) -> list[dict]:
        colors = [
            ("RED", "red", "r"),
            ("GREEN", "green", "g"),
            ("BLUE", "blue", "b"),
            ("YELLOW", "goldenrod", "y"),
        ]
        trials = []
        for _ in range(28):
            congruent = random.random() < 0.4
            word, display_color, correct_key = random.choice(colors)
            if congruent:
                displayed_word = word
                displayed_color = display_color
                expected_key = correct_key
            else:
                displayed_word, _, _ = random.choice([color for color in colors if color[0] != word])
                displayed_color = display_color
                expected_key = correct_key
            trials.append(
                {
                    "word": displayed_word,
                    "color_name": display_color,
                    "correct_key": expected_key,
                    "congruent": congruent,
                }
            )
        return trials

    def _start_stroop_task(self) -> None:
        self._clear()
        self.stroop_response_window_ms = TASK_TIMING["stroop"]["response_window_ms"]
        self.stroop_feedback_delay_ms = TASK_TIMING["stroop"]["feedback_ms"]
        self.stroop_trial_token = 0
        self._add_title("Selective Attention")
        self.stroop_progress = tk.Label(self.frame, text="", bg="#101820", fg="#78C6A3", font=("Segoe UI", 13))
        self.stroop_progress.pack(pady=(0, 8))
        self.stroop_word = tk.Label(self.frame, text="", bg="#101820", fg="#F2F7F2", font=("Segoe UI Black", 48))
        self.stroop_word.pack(pady=70)
        self.stroop_mapping = self._add_body("R=RED   G=GREEN   B=BLUE   Y=YELLOW", font_size=16)
        self.stroop_feedback = self._add_body(
            f"Respond to the font color only. You have up to {self._seconds_text(self.stroop_response_window_ms)}.",
            font_size=16,
        )
        self.stroop_trials = self._build_stroop_trials()
        self.stroop_index = 0
        self.stroop_phase = "idle"
        self.stroop_start_ts = 0.0
        self.active_key_handler = self._handle_stroop_key
        self._present_stroop_trial()

    def _present_stroop_trial(self) -> None:
        if self.stroop_index >= len(self.stroop_trials):
            self._complete_assessment()
            return
        self.stroop_trial_token += 1
        trial = self.stroop_trials[self.stroop_index]
        self.stroop_phase = "live"
        self.stroop_progress.config(text=f"Trial {self.stroop_index + 1} of {len(self.stroop_trials)}")
        self.stroop_word.config(text=trial["word"], fg=trial["color_name"])
        self.stroop_feedback.config(
            text=f"R / G / B / Y - up to {self._seconds_text(self.stroop_response_window_ms)}",
            fg="#D8E4D6",
        )
        self.stroop_start_ts = time.perf_counter()
        trial_token = self.stroop_trial_token
        self._schedule(
            self.stroop_response_window_ms,
            lambda token=trial_token: self._stroop_timeout(token),
        )

    def _record_stroop_trial(self, pressed_key: str | None, missed: bool) -> None:
        trial = dict(self.stroop_trials[self.stroop_index])
        response_ms = None if missed else round((time.perf_counter() - self.stroop_start_ts) * 1000.0, 2)
        trial.update(
            {
                "pressed_key": pressed_key,
                "response_ms": response_ms,
                "response_seconds": round(response_ms / 1000.0, 4) if response_ms is not None else None,
                "response_window_ms": self.stroop_response_window_ms,
                "missed": missed,
                "correct": (pressed_key == trial["correct_key"]) and not missed,
            }
        )
        self.results["stroop"]["trials"].append(trial)
        self.stroop_index += 1
        self.stroop_phase = "feedback"

    def _handle_stroop_key(self, event: tk.Event) -> None:
        pressed_key = event.keysym.lower()
        if self.stroop_phase != "live" or pressed_key not in {"r", "g", "b", "y"}:
            return
        self._record_stroop_trial(pressed_key, missed=False)
        last_trial = self.results["stroop"]["trials"][-1]
        if last_trial["correct"]:
            self.stroop_feedback.config(
                text=f"Correct - responded in {self._format_response_time(last_trial['response_ms'])}.",
                fg="#78C6A3",
            )
        else:
            self.stroop_feedback.config(
                text=f"Incorrect - responded in {self._format_response_time(last_trial['response_ms'])}.",
                fg="#F66B6B",
            )
        self._schedule(self.stroop_feedback_delay_ms, self._present_stroop_trial)

    def _stroop_timeout(self, trial_token: int) -> None:
        if trial_token != self.stroop_trial_token or self.stroop_phase != "live":
            return
        self._record_stroop_trial(pressed_key=None, missed=True)
        self.stroop_feedback.config(
            text=f"No response within {self._seconds_text(self.stroop_response_window_ms)}.",
            fg="#F2B35B",
        )
        self._schedule(self.stroop_feedback_delay_ms, self._present_stroop_trial)

    def _complete_assessment(self) -> None:
        finished_at = datetime.now()
        session_id = f"{self.participant_id}-{self.started_at.strftime('%Y%m%d-%H%M%S')}"
        self.session = {
            "session_id": session_id,
            "participant_id": self.participant_id,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "duration_seconds": int((finished_at - self.started_at).total_seconds()),
            "baseline_session": self.baseline_session,
            "questionnaire": self.questionnaire,
            "tasks": self.results,
        }
        self._show_message(
            title="Assessment Complete",
            body="The behavioral task battery is complete. Close this window to write the session report and summary.",
            button_text="Close",
            on_continue=self._close_after_completion,
        )

    def _close_after_completion(self) -> None:
        self.root.quit()
        self.root.destroy()


def run_interactive_assessment(
    participant_id: str,
    questionnaire: dict | None = None,
    baseline_session: bool = False,
) -> dict | None:
    return AssessmentApp(
        participant_id=participant_id,
        questionnaire=questionnaire,
        baseline_session=baseline_session,
    ).run()
