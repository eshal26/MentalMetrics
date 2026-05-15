const SESSION_KEY = "neuroXplain_sessions";
const THEME_KEY = "neuroXplain_theme";

export function loadSessions() {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (error) {
    return [];
  }
}

export function saveSessions(sessions) {
  localStorage.setItem(SESSION_KEY, JSON.stringify(sessions));
}

export function upsertSession(session) {
  const existing = loadSessions();
  const next = [session, ...existing.filter((item) => item.id !== session.id)];
  saveSessions(next);
  return next;
}

export function deleteSession(sessionId) {
  const next = loadSessions().filter((session) => session.id !== sessionId);
  saveSessions(next);
  return next;
}

export function clearSessions() {
  saveSessions([]);
  return [];
}

export function loadTheme() {
  return localStorage.getItem(THEME_KEY) || "light";
}

export function saveTheme(theme) {
  localStorage.setItem(THEME_KEY, theme);
}
