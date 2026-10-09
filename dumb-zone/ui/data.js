// Every number the page shows comes from here or from data/codex-sessions.json (measured on this laptop).

// NoLiMa results table, copied exactly from https://github.com/adobe-research/NoLiMa (README, "Results").
// base = base score (accuracy at 250, 500 and 1K tokens); null = not evaluated ("--" in the table).
// label = where the line's name sits (point index `at`, pixel offsets), so names don't collide.
export const NOLIMA_LENGTHS = [1, 2, 4, 8, 16, 32, 64, 128].map(k => k * 1000);
export const NOLIMA = [
  { id: 'gpt-4o', name: 'GPT-4o', claimed: '128K', effective: '8K', base: 99.3,
    scores: [98.1, 98.0, 95.7, 89.2, 81.6, 69.7, 62.4, 56.0] },
  { id: 'gpt-4.1', name: 'GPT-4.1', claimed: '1M', effective: '16K', base: 97.0,
    scores: [95.6, 95.2, 91.7, 87.5, 84.9, 79.8, 69.7, 64.7] },
  { id: 'gemini-2.0-flash', name: 'Gemini 2.0 Flash', claimed: '1M', effective: '4K', base: 89.4,
    scores: [87.7, 87.5, 77.9, 64.7, 48.2, 41.0, 33.0, 16.4] },
  { id: 'claude-3.5-sonnet', name: 'Claude 3.5 Sonnet', claimed: '200K', effective: '4K', base: 87.6,
    scores: [85.4, 84.0, 77.6, 61.7, 45.7, 29.8, null, null], label: { dy: 0 } },
  { id: 'llama-4-scout', name: 'Llama 4 Scout', claimed: '10M', effective: '1K', base: 81.7,
    scores: [72.3, 61.8, 50.8, 35.5, 26.9, 21.6, null, null], label: { at: 2, dx: 14, dy: 82 } },
];

// Fixed overhead before you type. Measured medians come from data/codex-sessions.json at runtime.
// Bar order, left to right. `legend` is the reading order in the 3-column legend.
export const SEGMENTS = [
  { id: 'system', label: 'System prompt', color: 'var(--s-system)', legend: 0 },
  { id: 'tools', label: 'Tool definitions', color: 'var(--s-tools)', legend: 1 },
  { id: 'mcp', label: 'MCP (GitHub)', color: 'var(--s-mcp)', legend: 2 },
  { id: 'memory', label: 'Memory files', color: 'var(--s-memory)', legend: 3 },
  { id: 'summary', label: 'Summary', color: 'var(--s-summary)', legend: 7 },
  { id: 'read', label: 'Files read', color: 'var(--s-read)', legend: 4 },
  { id: 'tool', label: 'Tool outputs', color: 'var(--s-tool)', legend: 5 },
  { id: 'chat', label: 'Chat history', color: 'var(--s-chat)', legend: 6 },
];

// Cited: Anthropic, "Introducing advanced tool use on the Claude Developer Platform" (Nov 2025):
// "GitHub: 35 tools (~26K tokens)".
export const MCP_GITHUB_TOKENS = 26000;

// Zones: Dex Horthy (HumanLayer). Quality slips past ~40% ("the dumb zone"); ACE-FCA keeps utilization at 40–60%.
export const ZONES = { dumb: 0.4, deep: 0.6 };

export const WINDOWS = [
  { tokens: 128000, label: '128K' },
  { tokens: 200000, label: '200K' },
  { tokens: 1000000, label: '1M' },
];

// ILLUSTRATION ONLY: the pattern Sinha et al. 2025 report, drawn by hand. Not measured, no rates implied.
// Turns listed are the wrong ones, over 30 turns. Both chats share the same early mistake at turn 3.
export const ILLUSTRATION = {
  turns: 30,
  mistakeAt: 3,
  freshFrom: 4,
  sameChatWrong: [3, 9, 14, 18, 21, 23, 25, 26, 28, 29],
  freshChatWrong: [3, 17],
};

export const SOURCES = {
  nolima: 'https://github.com/adobe-research/NoLiMa',
  sinha: 'https://arxiv.org/abs/2509.09677',
  anthropicTools: 'https://www.anthropic.com/engineering/advanced-tool-use',
  acefca: 'https://github.com/humanlayer/advanced-context-engineering-for-coding-agents/blob/main/ace-fca.md',
};
