# stream_client.py
# This script runs on the MacBook

import cv2
import socket
import pickle
import struct

# --- Configuration ---
# Use '0.0.0.0' to listen on all available network interfaces
LISTEN_IP = '0.0.0.0'
PORT = 9999
BUFFER_SIZE = 65536 # A large buffer size for UDP packets

def main():
    """
    Receives a video stream over UDP and displays it.
    """
    # Set up the UDP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Bind the socket to the address and port to listen for incoming data
    client_socket.bind((LISTEN_IP, PORT))
    print(f"Listening for stream on {LISTEN_IP}:{PORT}")

    # Create a buffer for incoming frame data
    data = b""
    
    # Calculate the size of the packed frame length (unsigned 64-bit integer 'Q')
    payload_size = struct.calcsize("Q")

    try:
        while True:
            # Receive packets until we have at least the payload_size
            while len(data) < payload_size:
                packet = client_socket.recv(BUFFER_SIZE)
                if not packet:
                    continue
                data += packet
            
            # Extract the packed message size
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            
            # Unpack the message size to know the full frame size
            msg_size = struct.unpack("Q", packed_msg_size)[0]
            
            # Receive packets until the full frame data is collected
            while len(data) < msg_size:
                data += client_socket.recv(BUFFER_SIZE)
                
            # Extract the complete frame data
            frame_data = data[:msg_size]
            data = data[msg_size:]
            
            # De-serialize the frame data back into an image
            frame = pickle.loads(frame_data)
            
            # Display the received frame
            cv2.imshow("Receiving Video Stream", frame)
            
            # Press 'q' on the keyboard to exit the video window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Receiving stopped by user.")
    finally:
        # Clean up resources
        client_socket.close()
        cv2.destroyAllWindows()
        print("Resources released.")

if __name__ == "__main__":
    main()