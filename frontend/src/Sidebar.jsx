import "./styles/Sidebar.css";

function Sidebar({
  chats,
  createChat,
  openChat,
  deleteChat,
  activeChatId,
  showSidebar,
}) {
  return (
    <div
      className={`sidebar ${
        showSidebar ? "" : "collapsed"
      }`}
    >

      <button
        className="new-chat"
        onClick={createChat}
      >
        + New Chat
      </button>

      <div className="chat-list">

        {chats.map((chat) => (
          <div
            key={chat.id}
            className={`chat-item ${
              activeChatId === chat.id
                ? "active"
                : ""
            }`}
            onClick={() => openChat(chat.id)}
          >

            <div className="chat-title">
              {chat.title}
            </div>

            <button
              className="delete-btn"
              onClick={(e) => {
                e.stopPropagation();
                deleteChat(chat.id);
              }}
            >
              🗑
            </button>

          </div>
        ))}

      </div>

    </div>
  );
}

export default Sidebar;