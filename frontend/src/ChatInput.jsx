import { useRef } from "react";
import "./styles/ChatInput.css";
function ChatInput({
  message,
  setMessage,
  sendMessage,
  handleFileUpload,
  selectedFiles,
  setSelectedFiles,
}) {

  const fileRef = useRef();

  return (
    <>
      {selectedFiles?.length > 0 && (
        <div className="file-preview">

          {selectedFiles?.map((file, index) => (
            <div key={index} className="preview-item">
              📎 {file.name}
            </div>
          ))}

          <button
            className="cancel-files"
            onClick={() => setSelectedFiles([])}
          >
            Cancel
          </button>

        </div>
      )}
      <div className="chat-input">

        <button
          className="attach-btn"
          onClick={() => fileRef.current.click()}
        >
          +
        </button>

        <input
          type="file"
          multiple
          hidden
          ref={fileRef}
          onChange={(e) =>
            handleFileUpload(e.target.files)
          }
        />

        <input
          value={message}
          placeholder="Message FitGPT..."
          onChange={(e) =>
            setMessage(e.target.value)
          }
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              sendMessage();
            }
          }}
        />

        <button onClick={sendMessage}>
          Send
        </button>

      </div>
    </>
  );

}

export default ChatInput;