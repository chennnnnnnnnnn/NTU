// survey-config.ts — single source of truth for the A/B comparison survey (v3).
//
// Each participant rates 6 sentences (2 from each set α/β/γ), each sentence
// gets one speaker (no speaker used >2× across the 6), and each sentence is
// asked as 3 pairwise A/B comparisons → 18 comparison items + 2 attention
// checks. Answers are a 1–5 bipolar scale (1 = A clearly, 5 = B clearly).

// ── Sampling ──────────────────────────────────────────────────────────
export const SETS = ["alpha", "beta", "gamma"] as const;
export const SENTENCES_PER_SET = 2;          // 2 per set → 6 sentences
export const MAX_SPEAKER_USES = 2;           // no speaker more than twice

export const SPEAKERS = [
  "b09207084",
  "b11901052",
  "r12323023",
  "r14525071",
  "r14922009",
  "r14922a21",
];

// Sentences to exclude from the pool (per analysis recommendation).
export const EXCLUDED_SENTENCES = ["gamma5", "beta6"];

/** All candidate sentence codes for a set, minus excluded ones. */
export function sentencePool(set: string): string[] {
  const all = [1, 2, 3, 4, 5, 6].map((n) => `${set}${n}`);
  return all.filter((code) => !EXCLUDED_SENTENCES.includes(code));
}

// ── Comparison questions (3 per sentence) ───────────────────────────────
// Each plays two clips (the two conditions) as A and B, asks one question.
export type QuestionType = "q1_accent" | "q2_human" | "q3_natural";

export type ComparisonQuestion = {
  type: QuestionType;
  /** the two conditions compared; order on screen is randomised per item */
  pair: [string, string];
  prompt: string;
  /** suffix used for the two end-labels: "A {endLabel}" … "B {endLabel}" */
  endLabel: string;
};

export const COMPARISON_QUESTIONS: ComparisonQuestion[] = [
  { type: "q1_accent",  pair: ["c2", "c3b"], prompt: "哪一段聽起來更像母語者？",   endLabel: "更像母語者" },
  { type: "q2_human",   pair: ["c3a", "c3b"], prompt: "哪一段聽起來比較像真人說話？", endLabel: "更像真人" },
  { type: "q3_natural", pair: ["c2", "c3a"], prompt: "哪一段聽起來比較自然？",     endLabel: "更自然" },
];

/** 1–5 bipolar scale. */
export const SCALE_MIN = 1;
export const SCALE_MAX = 5;

// ── Attention checks (shown at the start of the rating sequence) ────────
// Each is a 1–5 scale where the respondent must pick `expected`.
export const ATTENTION_CHECKS = [
  { expected: 2, prompt: "注意力檢查：請在此題勾選「2」。" },
  { expected: 5, prompt: "注意力檢查：請在此題勾選「5」。" },
];

/** Show the target sentence text alongside the players.
 *  Off: the personalised clips' true transcript isn't available, and for an
 *  A/B listening comparison the text is unnecessary (and could bias). */
export const SHOW_SENTENCE = false;

/** For display copy on the landing page. */
export const TOTAL_QUESTIONS =
  SETS.length * SENTENCES_PER_SET * COMPARISON_QUESTIONS.length; // 18

// ── Start-page selector: only "第幾人" remains (identifier) ──────────────
export type BatchSelector = {
  key: "person_num";
  label: string;
  options: { value: string; label: string }[];
};

export const BATCH_SELECTORS: BatchSelector[] = [
  {
    key: "person_num",
    label: "第幾人",
    options: Array.from({ length: 10 }, (_, i) => ({
      value: String(i + 1),
      label: String(i + 1),
    })),
  },
];

// ── Demographic fields (profile page) ───────────────────────────────────
export type ProfileField = {
  key: "age_bracket" | "gender" | "native_language" | "english_level";
  label: string;
  options: { value: string; label: string }[];
};

export const PROFILE_FIELDS: ProfileField[] = [
  {
    key: "age_bracket",
    label: "年齡層",
    options: [
      { value: "18-24", label: "18–24" },
      { value: "25-34", label: "25–34" },
      { value: "35-44", label: "35–44" },
      { value: "45+", label: "45 以上" },
    ],
  },
  {
    key: "gender",
    label: "性別",
    options: [
      { value: "male", label: "男" },
      { value: "female", label: "女" },
      { value: "prefer_not", label: "不願透露" },
    ],
  },
  {
    key: "native_language",
    label: "母語",
    options: [
      { value: "mandarin", label: "中文" },
      { value: "english", label: "英文" },
      { value: "other", label: "其他" },
    ],
  },
  {
    key: "english_level",
    label: "英語程度（CEFR 自評）",
    options: [
      { value: "A1", label: "A1 入門" },
      { value: "A2", label: "A2 基礎" },
      { value: "B1", label: "B1 中級" },
      { value: "B2", label: "B2 中高級" },
      { value: "C1", label: "C1 高級" },
      { value: "C2", label: "C2 精通" },
    ],
  },
];
