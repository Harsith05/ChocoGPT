// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";
import { getStorage } from "firebase/storage";

// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyAwfrX6fgKwJnAWS-hmGd4sAAkUHk0uxbg",
  authDomain: "fitgpt-63143.firebaseapp.com",
  projectId: "fitgpt-63143",
  storageBucket: "fitgpt-63143.firebasestorage.app",
  messagingSenderId: "471167147223",
  appId: "1:471167147223:web:b7aeb090b98bf51a7aa492",
};


// Initialize Firebase
const app = initializeApp(firebaseConfig);

export const db = getFirestore(app);
export const auth = getAuth(app);
export const storage = getStorage(app);