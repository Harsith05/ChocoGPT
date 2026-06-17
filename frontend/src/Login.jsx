import { useState } from "react";
import "./styles/Login.css";

import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
} from "firebase/auth";

import { auth } from "./firebase/config";
import { useNavigate } from "react-router-dom";

function Login() {
  const navigate = useNavigate();

  const [isLogin, setIsLogin] = useState(true);

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleLogin = async () => {
    try {
      await signInWithEmailAndPassword(
        auth,
        email,
        password
      );

      navigate("/chat");
    } catch (error) {
      alert(error.message);
    }
  };

  const handleSignup = async () => {
    try {
      await createUserWithEmailAndPassword(
        auth,
        email,
        password
      );

      alert("Account Created Successfully");

      navigate("/chat");

    } catch (error) {
      alert(error.message);
    }
  };

  return (
    <div className="login-page">

      <div className="login-card">

        <h1 className="auth-title">
          FitGPT
        </h1>

        <p className="auth-subtitle">
          {isLogin
            ? "Welcome back to your AI Fitness Assistant"
            : "Create your FitGPT account"}
        </p>

        <input
          type="email"
          placeholder="Enter your email"
          value={email}
          onChange={(e) =>
            setEmail(e.target.value)
          }
        />

        <input
          type="password"
          placeholder="Enter your password"
          value={password}
          onChange={(e) =>
            setPassword(e.target.value)
          }
        />

        {isLogin ? (
          <>
            <button
              className="primary-btn"
              onClick={handleLogin}
            >
              Login
            </button>

            <p className="switch-text">
              Don't have an account?{" "}
              <span
                onClick={() =>
                  setIsLogin(false)
                }
              >
                Sign Up
              </span>
            </p>
          </>
        ) : (
          <>
            <button
              className="primary-btn"
              onClick={handleSignup}
            >
              Create Account
            </button>

            <p className="switch-text">
              Already have an account?{" "}
              <span
                onClick={() =>
                  setIsLogin(true)
                }
              >
                Login
              </span>
            </p>
          </>
        )}

      </div>

    </div>
  );
}

export default Login;