/* Browser-side timing constants.
   These deliberately mirror TASK_TIMING in src/cognitive_load/config.py so a
   session taken in the browser uses the same pacing as the desktop battery.
   If you change one, change both. */

export const TASK_TIMING = {
  reaction_time: {
    response_window_ms: 60000,
    feedback_ms: 900,
    pre_signal_delay_ms: [1400, 3000],
  },
  task_switching: {
    response_window_ms: 60000,
    feedback_ms: 1000,
  },
  working_memory: {
    pre_load_ms: 500,
    digit_display_ms: 900,
    digit_gap_ms: 350,
    post_load_ms: 300,
    recall_window_ms: 60000,
    feedback_ms: 1200,
  },
  stroop: {
    response_window_ms: 60000,
    feedback_ms: 1000,
  },
};

/** Presentation gap before each task's first trial. */
export const TASK_INTRO_MS = 1600;
