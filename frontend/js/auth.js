// auth.js - handle sign in / sign up toggle and authentication
import { supabase } from "./supabase.js";
import { N8N_WEBHOOK_URL } from "./config.js"; // ensures config loads

const form = document.getElementById("auth-form");
const emailInput = document.getElementById("email");
const passwordInput = document.getElementById("password");
const authButton = document.getElementById("auth-button");
const toggleLink = document.getElementById("toggle-link");
const toggleText = document.getElementById("toggle-text");
const messageDiv = document.getElementById("message");

let isSignUp = false; // start in Sign In mode

function setLoading(loading) {
  if (loading) {
    authButton.disabled = true;
    authButton.textContent = isSignUp ? "Signing up..." : "Signing in...";
    const spinner = document.createElement("span");
    spinner.className = "loading-spinner";
    authButton.appendChild(spinner);
  } else {
    authButton.disabled = false;
    authButton.textContent = isSignUp ? "Sign Up" : "Sign In";
  }
}

function showMessage(text, type) {
  messageDiv.textContent = text;
  messageDiv.className = `message ${type}`;
  messageDiv.classList.remove("hidden");
}

function clearMessage() {
  messageDiv.textContent = "";
  messageDiv.className = "message hidden";
}

function updateToggle() {
  if (isSignUp) {
    authButton.textContent = "Sign Up";
    toggleText.textContent = "Already have an account?";
    toggleLink.textContent = "Sign In";
  } else {
    authButton.textContent = "Sign In";
    toggleText.textContent = "Don't have an account?";
    toggleLink.textContent = "Sign Up";
  }
}

toggleLink.addEventListener("click", (e) => {
  e.preventDefault();
  isSignUp = !isSignUp;
  clearMessage();
  updateToggle();
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearMessage();
  setLoading(true);
  const email = emailInput.value.trim();
  const password = passwordInput.value.trim();
  try {
    let result;
    if (isSignUp) {
      result = await supabase.auth.signUp({ email, password });
    } else {
      result = await supabase.auth.signInWithPassword({ email, password });
    }
    const { error } = result;
    if (error) {
      showMessage(error.message, "error");
    } else {
      // successful login or signup
      window.location.href = "dashboard.html";
    }
  } catch (err) {
    showMessage(err.message, "error");
  } finally {
    setLoading(false);
  }
});

// redirect if already logged in
supabase.auth.getSession().then(({ data: { session } }) => {
  if (session) {
    window.location.href = "dashboard.html";
  }
});

updateToggle();
