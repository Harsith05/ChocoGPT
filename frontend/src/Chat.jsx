import { useEffect, useState } from "react";
import {
  collection,
  addDoc,
  getDocs,
  query,
  orderBy,
  serverTimestamp,
  deleteDoc,
  doc,
  updateDoc,
} from "firebase/firestore";

import { signOut } from "firebase/auth";
import { db, auth } from "./firebase/config";

import Sidebar from "./Sidebar";
import MessageBubble from "./MessageBubble";
import ChatInput from "./ChatInput";

import "./styles/Chat.css";

function Chat() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [chats, setChats] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [showSidebar, setShowSidebar] = useState(true);
  const [selectedFiles, setSelectedFiles] = useState([]);

  useEffect(() => {
    const unsub = auth.onAuthStateChanged((user) => {
      if (user) {
        loadChats(user.uid);
      }
    });

    return () => unsub();
  }, []);

  const loadChats = async (uid) => {
    const q = query(
      collection(db, "users", uid, "chats"),
      orderBy("createdAt", "desc")
    );

    const snap = await getDocs(q);

    const data = snap.docs.map((d) => ({
      id: d.id,
      ...d.data(),
    }));

    setChats(data);

    if (data.length > 0) {
      openChat(data[0].id, uid);
    }
  };

  const createChat = async () => {
    const uid = auth.currentUser.uid;

    const docRef = await addDoc(
      collection(db, "users", uid, "chats"),
      {
        title: "New Chat",
        createdAt: serverTimestamp(),
      }
    );

    const newChat = {
      id: docRef.id,
      title: "New Chat",
    };

    setChats((prev) => [newChat, ...prev]);
    openChat(docRef.id, uid);
  };

  const openChat = async (chatId, uid = auth.currentUser?.uid) => {
    if (!chatId || !uid) return;

    setCurrentChatId(chatId);
    setMessages([]);

    const q = query(
      collection(db, "users", uid, "chats", chatId, "messages"),
      orderBy("createdAt")
    );

    const snap = await getDocs(q);

    setMessages(snap.docs.map((d) => d.data()));
  };

  const updateChatTitle = async (chatId, text) => {
    const uid = auth.currentUser.uid;

    const shortTitle =
      text.length > 25 ? text.substring(0, 25) + "..." : text;

    await updateDoc(
      doc(db, "users", uid, "chats", chatId),
      {
        title: shortTitle,
      }
    );

    setChats((prev) =>
      prev.map((c) =>
        c.id === chatId ? { ...c, title: shortTitle } : c
      )
    );
  };

  const deleteChat = async (chatId) => {
    const uid = auth.currentUser.uid;

    await deleteDoc(doc(db, "users", uid, "chats", chatId));

    const updated = chats.filter((c) => c.id !== chatId);
    setChats(updated);

    if (currentChatId === chatId) {
      setMessages([]);
      setCurrentChatId(null);

      if (updated.length > 0) {
        openChat(updated[0].id, uid);
      }
    }
  };

  const sendMessage = async () => {
    const uid = auth.currentUser.uid;

    let chatId = currentChatId;

    if (!message.trim() && selectedFiles.length === 0) return;

    if (!chatId) {
      const docRef = await addDoc(
        collection(db, "users", uid, "chats"),
        {
          title: message || "Image chat",
          createdAt: serverTimestamp(),
        }
      );

      chatId = docRef.id;
      setCurrentChatId(chatId);

      const newChat = { id: chatId, title: message || "Image chat" };
      setChats((prev) => [newChat, ...prev]);
    }

    // Pull out the first image among selected files (cake photo).
    // Any other non-image files keep the old placeholder behavior.
    const imageFile = selectedFiles.find((f) =>
      f.type?.startsWith("image/")
    );
    const otherFiles = selectedFiles.filter((f) => f !== imageFile);

    if (imageFile) {
      // NOTE: Firebase Storage requires the Blaze (pay-as-you-go) plan —
      // on the free Spark plan, Storage calls fail with CORS/402/403
      // errors. We skip Storage entirely and use a local blob URL just
      // to show the thumbnail in this browser tab/session. It is NOT
      // persisted: it won't survive a refresh or show on another device,
      // and we don't send it to Firestore (a blob: URL is meaningless
      // outside this tab).
      const localPreviewUrl = URL.createObjectURL(imageFile);

      const imageMsg = {
        role: "user",
        isFile: true,
        isImage: true,
        fileName: imageFile.name,
        fileUrl: localPreviewUrl,
        text: message || "📷 Image",
      };

      setMessages((prev) => [...prev, imageMsg]);

      // Save a lightweight placeholder to Firestore history (no blob URL,
      // since it wouldn't resolve in a future session anyway).
      await addDoc(
        collection(db, "users", uid, "chats", chatId, "messages"),
        {
          role: "user",
          isFile: true,
          isImage: true,
          fileName: imageFile.name,
          text: message || "📷 Image",
          createdAt: serverTimestamp(),
        }
      );
    }

    if (otherFiles.length > 0) {
      for (const file of otherFiles) {
        const fileMsg = {
          role: "user",
          isFile: true,
          fileName: file.name,
          text: `📎 ${file.name}`,
        };

        setMessages((prev) => [...prev, fileMsg]);

        await addDoc(
          collection(db, "users", uid, "chats", chatId, "messages"),
          {
            ...fileMsg,
            createdAt: serverTimestamp(),
          }
        );
      }
    }

    setSelectedFiles([]);

    const currentText = message;

    // Only show a separate text bubble if there's no image attached
    // (the image bubble above already carries the typed caption).
    if (message.trim() && !imageFile) {
      const userMsg = {
        role: "user",
        text: message,
      };

      setMessages((prev) => [...prev, userMsg]);

      await addDoc(
        collection(db, "users", uid, "chats", chatId, "messages"),
        {
          ...userMsg,
          createdAt: serverTimestamp(),
        }
      );
    }

    setMessage("");

    const chat = chats.find((c) => c.id === chatId);

    if (chat?.title === "New Chat") {
      updateChatTitle(chatId, currentText || "Image chat");
    }

    try {
      let response;

      if (imageFile) {
        // Image present -> hit the vision + RAG endpoint.
        // Qwen2.5-VL identifies the cake, then the same hybrid_retrieve
        // pipeline used for text queries runs on the detected name.
        const formData = new FormData();
        formData.append("file", imageFile);
        formData.append("query", currentText);

        response = await fetch("http://localhost:8000/chat-image", {
          method: "POST",
          body: formData,
        });
      } else {
        response = await fetch("http://localhost:8000/chat", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            query: currentText,
          }),
        });
      }

      if (!response.ok) {
        throw new Error("Backend error");
      }

      const data = await response.json();

      // FIX: use data.image_urls (plural) to match the API response,
      // and store as imageUrls to match the MessageBubble prop name.
      // Also handle the no-match case where data.recipe is null/undefined
      // and the API returns { answer: "...", image_url: null } instead.
      // detected_cake_type is only present on /chat-image responses.
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

      await addDoc(
        collection(db, "users", uid, "chats", chatId, "messages"),
        {
          ...botMsg,
          createdAt: serverTimestamp(),
        }
      );
    } catch (error) {
      console.error(error);

      const errorMsg = {
        role: "assistant",
        text: "Failed to connect to AI backend",
      };

      setMessages((prev) => [...prev, errorMsg]);

      await addDoc(
        collection(db, "users", uid, "chats", chatId, "messages"),
        {
          ...errorMsg,
          createdAt: serverTimestamp(),
        }
      );
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
          <button
            className="sidebar-toggle"
            onClick={() => setShowSidebar(!showSidebar)}
          >
            ☰
          </button>
          <div className="title">FitGPT</div>

          <button onClick={handleLogout} className="logout-btn">
            Logout
          </button>
        </div>

        <div className="messages">
          {messages.length === 0 && (
            <div className="empty">Start a conversation...</div>
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
        </div>

        <ChatInput
          message={message}
          setMessage={setMessage}
          sendMessage={sendMessage}
          handleFileUpload={handleFileUpload}
          selectedFiles={selectedFiles}
          setSelectedFiles={setSelectedFiles}
        />
      </div>
    </div>
  );
}

export default Chat;
