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

          {selectedFiles?.map((file, index) => {
            const isImage = file.type?.startsWith("image/");
            const previewUrl = isImage ? URL.createObjectURL(file) : null;

            return (
              <div key={index} className="preview-item">
                {isImage ? (
                  <img
                    src={previewUrl}
                    alt={file.name}
                    style={{
                      width: "48px",
                      height: "48px",
                      objectFit: "cover",
                      borderRadius: "6px",
                      marginRight: "6px",
                      verticalAlign: "middle",
                    }}
                  />
                ) : (
                  "📎 "
                )}
                {file.name}
              </div>
            );
          })}

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
          accept="image/*,.jpg,.jpeg,.png,.webp,.gif,.bmp"
          ref={fileRef}
          onChange={(e) => {
            handleFileUpload(e.target.files);
            // Reset the input's value so selecting the SAME file again
            // later still fires onChange. Without this, the browser
            // treats an unchanged file list as "nothing changed" and
            // silently does nothing on the next pick.
            e.target.value = "";
          }}
        />

        <input
          value={message}
          placeholder="Message ChocoGPT..."
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