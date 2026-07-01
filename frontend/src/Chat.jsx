import { useEffect, useState, useRef } from "react";
import { signOut } from "firebase/auth";
import { auth } from "./firebase/config";

import Sidebar     from "./Sidebar";
import MessageBubble from "./MessageBubble";
import LoadingBubble from "./LoadingBubble";
import ChatInput   from "./ChatInput";
import "./styles/Chat.css";

// All chat/message/image persistence now goes through the FastAPI + MongoDB
// backend instead of Firestore. `id` on chat/message objects below is the
// Mongo ObjectId string (serialized to `_id` -> `id` on the way in).
const API_BASE = "http://localhost:8000";

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
    const unsub = auth.onAuthStateChanged(async (user) => {
      if (user) {
        // Upsert the user doc in Mongo (mirrors the old Firebase-uid-as-doc-id pattern)
        await ensureUser(user.uid, user.email || "", user.displayName || "");
        loadChats(user.uid);
      }
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
  // Maps a message document returned by the backend (Mongo shape)
  // into the flat prop shape MessageBubble/chat state expects.
  // -------------------------------------------------------
  const mapMessageDoc = (d) => {
    const hasImageRef = d.is_file && Array.isArray(d.image_refs) && d.image_refs.length > 0;
    return {
      id:      d._id,
      role:    d.role,
      text:    d.text || "",
      recipe:  d.recipe || undefined,
      imageUrls: d.image_urls || [],
      isFile:  !!d.is_file,
      fileName: d.file_name || "",
      isImage: hasImageRef,
      fileUrl: hasImageRef ? `${API_BASE}/images/${d.image_refs[0]}` : undefined,
      detectedCakeType: d.detected_cake_type || null,
    };
  };

  // -------------------------------------------------------
  // Backend REST helpers  (MongoDB, via FastAPI)
  // -------------------------------------------------------
  const ensureUser = async (uid, email, displayName) => {
    try {
      await fetch(`${API_BASE}/api/users/ensure`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ uid, email, display_name: displayName }),
      });
    } catch (err) {
      console.error("ensureUser failed:", err);
    }
  };

  const loadChats = async (uid) => {
    const res  = await fetch(`${API_BASE}/api/chats/${uid}`);
    const data = await res.json();
    const mapped = data.map((c) => ({ id: c._id, title: c.title }));
    setChats(mapped);
    if (mapped.length > 0) openChat(mapped[0].id);
  };

  const createChat = async (title = "New Chat") => {
    const uid = auth.currentUser?.uid;
    if (!uid) return null;
    const res  = await fetch(`${API_BASE}/api/chats`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: uid, title }),
    });
    const chat = await res.json();
    const newChat = { id: chat._id, title: chat.title };
    setChats((prev) => [newChat, ...prev]);
    openChat(newChat.id);
    return newChat.id;
  };

  const openChat = async (chatId) => {
    if (!chatId) return;
    setCurrentChatId(chatId);
    setMessages([]);
    const res  = await fetch(`${API_BASE}/api/chats/${chatId}/messages`);
    const data = await res.json();
    setMessages(data.map(mapMessageDoc));
  };

  const updateChatTitle = async (chatId, text) => {
    const shortTitle = text.length > 25 ? text.substring(0, 25) + "..." : text;
    await fetch(`${API_BASE}/api/chats/${chatId}/title`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: shortTitle }),
    });
    setChats((prev) => prev.map((c) => c.id === chatId ? { ...c, title: shortTitle } : c));
  };

  const deleteChat = async (chatId) => {
    await fetch(`${API_BASE}/api/chats/${chatId}`, { method: "DELETE" });
    const updated = chats.filter((c) => c.id !== chatId);
    setChats(updated);
    if (currentChatId === chatId) {
      setMessages([]);
      setCurrentChatId(null);
      if (updated.length > 0) openChat(updated[0].id);
    }
  };

  // Persists a plain text/recipe message (no attached file) to Mongo.
  const persistMessage = async (chatId, uid, msg) => {
    try {
      await fetch(`${API_BASE}/api/chats/${chatId}/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id:            uid,
          role:                msg.role,
          text:                msg.text || "",
          recipe:              msg.recipe || null,
          image_urls:          msg.imageUrls || [],
          is_file:             false,
          file_name:           "",
          detected_cake_type:  msg.detectedCakeType || null,
        }),
      });
    } catch (err) {
      console.error("persistMessage failed:", err);
    }
  };

  // Uploads a file's bytes to GridFS and persists the message that
  // references it, in one call. Returns the saved message doc so the
  // caller can swap the local blob preview for the permanent /images/ URL.
  const persistFileMessage = async (chatId, uid, file, text) => {
    const formData = new FormData();
    formData.append("user_id", uid);
    formData.append("text", text || "");
    formData.append("file", file);

    const res = await fetch(`${API_BASE}/api/chats/${chatId}/messages/file`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error("Failed to persist file message");
    return res.json();
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
      chatId = await createChat(message || "Image chat");
      if (!chatId) return;
    }

    const imageFile  = selectedFiles.find((f) => f.type?.startsWith("image/"));
    const otherFiles = selectedFiles.filter((f) => f !== imageFile);

    // Show image bubble — optimistic local preview first, then swap in the
    // permanent GridFS-backed URL once the upload finishes.
    if (imageFile) {
      const localPreviewUrl = URL.createObjectURL(imageFile);
      const tempId = `local-${Date.now()}`;
      const imageMsg = {
        id: tempId,
        role: "user", isFile: true, isImage: true,
        fileName: imageFile.name, fileUrl: localPreviewUrl,
        text: message || "📷 Image",
      };
      setMessages((prev) => [...prev, imageMsg]);

      try {
        const saved = await persistFileMessage(chatId, uid, imageFile, message || "📷 Image");
        setMessages((prev) => prev.map((m) =>
          m.id === tempId ? { ...m, id: saved._id, fileUrl: `${API_BASE}${saved.file_url}` } : m
        ));
      } catch (err) {
        console.error("Image upload failed:", err);
      }
    }

    // Other files — reuses the same GridFS-backed endpoint (works for any file type).
    for (const file of otherFiles) {
      const tempId = `local-${Date.now()}-${file.name}`;
      const fileMsg = { id: tempId, role: "user", isFile: true, fileName: file.name, text: `📎 ${file.name}` };
      setMessages((prev) => [...prev, fileMsg]);
      try {
        const saved = await persistFileMessage(chatId, uid, file, `📎 ${file.name}`);
        setMessages((prev) => prev.map((m) => m.id === tempId ? { ...m, id: saved._id } : m));
      } catch (err) {
        console.error("File upload failed:", err);
      }
    }

    setSelectedFiles([]);

    const currentText = message;

    // Text bubble (only when no image)
    if (message.trim() && !imageFile) {
      const userMsg = { role: "user", text: message };
      setMessages((prev) => [...prev, userMsg]);
      await persistMessage(chatId, uid, userMsg);
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

        response = await fetch(`${API_BASE}/chat-image`, {
          method: "POST",
          body: formData,
        });
      } else {
        response = await fetch(`${API_BASE}/chat`, {
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
      await persistMessage(chatId, uid, botMsg);

    } catch (error) {
      console.error(error);
      const errorMsg = { role: "assistant", text: "Failed to connect to AI backend" };
      setMessages((prev) => [...prev, errorMsg]);
      await persistMessage(chatId, uid, errorMsg);
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
        createChat={() => createChat()}
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
              key={m.id || i}
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
