import { useEffect, useState, useRef } from "react";
import {
  collection, addDoc, getDocs, query,
  orderBy, serverTimestamp, deleteDoc, doc, updateDoc,
} from "firebase/firestore";
import { signOut } from "firebase/auth";
import { db, auth } from "./firebase/config";

import Sidebar     from "./Sidebar";
import MessageBubble from "./MessageBubble";
import LoadingBubble from "./LoadingBubble";
import ChatInput   from "./ChatInput";
import "./styles/Chat.css";

function Chat() {
  const [message, setMessage]           = useState("");
  const [messages, setMessages]         = useState([]);
  const [chats, setChats]               = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [showSidebar, setShowSidebar]   = useState(true);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [isLoading, setIsLoading]       = useState(false);
  const messagesEndRef                  = useRef(null);

  // Auto-scroll to bottom on new messages (including the loading bubble)
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  useEffect(() => {
    const unsub = auth.onAuthStateChanged((user) => {
      if (user) loadChats(user.uid);
    });
    return () => unsub();
  }, []);

  // -------------------------------------------------------
  // Build history payload — converts messages[] to plain text
  // summaries the backend can inject into Gemini's context.
  // Sends the FULL conversation (not just a recent slice) — the
  // backend owns windowing/summarization decisions so older context
  // isn't silently discarded before it even reaches the server.
  // -------------------------------------------------------
  const buildHistory = (msgs) => {
    return msgs
      .filter((m) => !m.isFile)                 // skip file bubbles
      .map((m) => ({
        role: m.role,
        text: m.recipe
          ? `[Recipe: ${m.recipe.recipe_name}${m.recipe.scaled ? " (scaled)" : ""}]`
          : (m.text || ""),
      }))
      .filter((m) => m.text);
  };

  // -------------------------------------------------------
  // Find the most recent assistant message that carried a
  // structured recipe. Sent to the backend so it can ground
  // follow-up questions ("only 2 eggs enough for this cake")
  // without re-running RAG retrieval.
  // -------------------------------------------------------
  const getLastRecipe = (msgs) => {
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i].role === "assistant" && msgs[i].recipe) {
        return msgs[i].recipe;
      }
    }
    return null;
  };

  // -------------------------------------------------------
  // Firestore helpers
  // -------------------------------------------------------
  const loadChats = async (uid) => {
    const q    = query(collection(db, "users", uid, "chats"), orderBy("createdAt", "desc"));
    const snap = await getDocs(q);
    const data = snap.docs.map((d) => ({ id: d.id, ...d.data() }));
    setChats(data);
    if (data.length > 0) openChat(data[0].id, uid);
  };

  const createChat = async () => {
    const uid    = auth.currentUser.uid;
    const docRef = await addDoc(collection(db, "users", uid, "chats"), {
      title: "New Chat", createdAt: serverTimestamp(),
    });
    const newChat = { id: docRef.id, title: "New Chat" };
    setChats((prev) => [newChat, ...prev]);
    openChat(docRef.id, uid);
  };

  const openChat = async (chatId, uid = auth.currentUser?.uid) => {
    if (!chatId || !uid) return;
    setCurrentChatId(chatId);
    setMessages([]);
    const q    = query(collection(db, "users", uid, "chats", chatId, "messages"), orderBy("createdAt"));
    const snap = await getDocs(q);
    setMessages(snap.docs.map((d) => d.data()));
  };

  const updateChatTitle = async (chatId, text) => {
    const uid        = auth.currentUser.uid;
    const shortTitle = text.length > 25 ? text.substring(0, 25) + "..." : text;
    await updateDoc(doc(db, "users", uid, "chats", chatId), { title: shortTitle });
    setChats((prev) => prev.map((c) => c.id === chatId ? { ...c, title: shortTitle } : c));
  };

  const deleteChat = async (chatId) => {
    const uid     = auth.currentUser.uid;
    await deleteDoc(doc(db, "users", uid, "chats", chatId));
    const updated = chats.filter((c) => c.id !== chatId);
    setChats(updated);
    if (currentChatId === chatId) {
      setMessages([]);
      setCurrentChatId(null);
      if (updated.length > 0) openChat(updated[0].id, uid);
    }
  };

  // -------------------------------------------------------
  // sendMessage — core logic
  // -------------------------------------------------------
  const sendMessage = async () => {
    const uid = auth.currentUser?.uid;
    if (!uid) return;
    if (isLoading) return; // guard against duplicate sends while waiting

    let chatId = currentChatId;
    if (!message.trim() && selectedFiles.length === 0) return;

    // Create chat if none active
    if (!chatId) {
      const docRef = await addDoc(collection(db, "users", uid, "chats"), {
        title: message || "Image chat", createdAt: serverTimestamp(),
      });
      chatId = docRef.id;
      setCurrentChatId(chatId);
      setChats((prev) => [{ id: chatId, title: message || "Image chat" }, ...prev]);
    }

    const imageFile  = selectedFiles.find((f) => f.type?.startsWith("image/"));
    const otherFiles = selectedFiles.filter((f) => f !== imageFile);

    // Show image bubble
    if (imageFile) {
      const localPreviewUrl = URL.createObjectURL(imageFile);
      const imageMsg = {
        role: "user", isFile: true, isImage: true,
        fileName: imageFile.name, fileUrl: localPreviewUrl,
        text: message || "📷 Image",
      };
      setMessages((prev) => [...prev, imageMsg]);
      await addDoc(collection(db, "users", uid, "chats", chatId, "messages"), {
        role: "user", isFile: true, isImage: true,
        fileName: imageFile.name, text: message || "📷 Image",
        createdAt: serverTimestamp(),
      });
    }

    // Other files
    for (const file of otherFiles) {
      const fileMsg = { role: "user", isFile: true, fileName: file.name, text: `📎 ${file.name}` };
      setMessages((prev) => [...prev, fileMsg]);
      await addDoc(collection(db, "users", uid, "chats", chatId, "messages"), {
        ...fileMsg, createdAt: serverTimestamp(),
      });
    }

    setSelectedFiles([]);

    const currentText = message;

    // Text bubble (only when no image)
    if (message.trim() && !imageFile) {
      const userMsg = { role: "user", text: message };
      setMessages((prev) => [...prev, userMsg]);
      await addDoc(collection(db, "users", uid, "chats", chatId, "messages"), {
        ...userMsg, createdAt: serverTimestamp(),
      });
    }

    setMessage("");

    const chat = chats.find((c) => c.id === chatId);
    if (chat?.title === "New Chat") updateChatTitle(chatId, currentText || "Image chat");

    // Snapshot current messages for history BEFORE adding bot reply
    const historySnapshot = [...messages];

    // ---- Call backend ----
    setIsLoading(true);
    try {
      let response;
      const history    = buildHistory(historySnapshot);
      const lastRecipe = getLastRecipe(historySnapshot);

      if (imageFile) {
        const formData = new FormData();
        formData.append("file", imageFile);
        formData.append("query", currentText);
        formData.append("history", JSON.stringify(history));
        formData.append("last_recipe", JSON.stringify(lastRecipe));

        response = await fetch("http://localhost:8000/chat-image", {
          method: "POST",
          body: formData,
        });
      } else {
        response = await fetch("http://localhost:8000/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: currentText, history, last_recipe: lastRecipe }),
        });
      }

      if (!response.ok) throw new Error("Backend error");

      const data = await response.json();

      // The static recipe-card template (Ingredients / Instructions /
      // Prep Time / Servings) only renders when data.recipe is present —
      // i.e. only on a fresh RAG retrieval. Off-topic deflections and
      // follow-up answers come back as plain `data.answer` text and use
      // the normal chat bubble instead.
      const botMsg = data.recipe
        ? {
            role: "assistant",
            recipe: data.recipe,
            imageUrls: data.image_urls || [],
            detectedCakeType: data.detected_cake_type || null,
          }
        : {
            role: "assistant",
            text: data.answer || "No matching recipe found.",
            detectedCakeType: data.detected_cake_type || null,
          };

      setMessages((prev) => [...prev, botMsg]);

      await addDoc(collection(db, "users", uid, "chats", chatId, "messages"), {
        ...botMsg, createdAt: serverTimestamp(),
      });

    } catch (error) {
      console.error(error);
      const errorMsg = { role: "assistant", text: "Failed to connect to AI backend" };
      setMessages((prev) => [...prev, errorMsg]);
      await addDoc(collection(db, "users", uid, "chats", chatId, "messages"), {
        ...errorMsg, createdAt: serverTimestamp(),
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = async () => {
    await signOut(auth);
    window.location.href = "/";
  };

  const handleFileUpload = (files) => {
    if (!files.length) return;
    setSelectedFiles(Array.from(files));
  };

  return (
    <div className="chat-container">
      <Sidebar
        chats={chats}
        createChat={createChat}
        openChat={openChat}
        deleteChat={deleteChat}
        activeChatId={currentChatId}
        showSidebar={showSidebar}
      />

      <div className="chat-main">
        <div className="topbar">
          <button className="sidebar-toggle" onClick={() => setShowSidebar(!showSidebar)}>☰</button>
          <div className="title">CocoaGPT</div>
          <button onClick={handleLogout} className="logout-btn">Logout</button>
        </div>

        <div className="messages">
          {messages.length === 0 && !isLoading && (
            <div className="empty">🍫 Ask me for any cake recipe...</div>
          )}
          {messages.map((m, i) => (
            <MessageBubble
              key={i}
              role={m.role}
              text={m.text}
              recipe={m.recipe}
              imageUrls={m.imageUrls}
              fileUrl={m.fileUrl}
              fileName={m.fileName}
              isFile={m.isFile}
              isImage={m.isImage}
              detectedCakeType={m.detectedCakeType}
            />
          ))}
          {isLoading && <LoadingBubble />}
          <div ref={messagesEndRef} />
        </div>

        <ChatInput
          message={message}
          setMessage={setMessage}
          sendMessage={sendMessage}
          handleFileUpload={handleFileUpload}
          selectedFiles={selectedFiles}
          setSelectedFiles={setSelectedFiles}
          isLoading={isLoading}
        />
      </div>
    </div>
  );
}

export default Chat;
