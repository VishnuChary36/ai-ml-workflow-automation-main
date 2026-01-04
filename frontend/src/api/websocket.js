const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

export class LogWebSocket {
  constructor(taskId, onMessage, onError = null, onClose = null) {
    this.taskId = taskId;
    this.onMessage = onMessage;
    this.onError = onError;
    this.onClose = onClose;
    this.ws = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.shouldReconnect = true;
    this.isConnecting = false;
    this.isDisconnected = false;
    this.hasConnected = false;
  }

  connect() {
    // Don't connect if already disconnected or connecting
    if (this.isDisconnected) {
      return;
    }
    
    if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.OPEN)) {
      return;
    }
    
    this.isConnecting = true;
    const url = `${WS_BASE_URL}/ws/logs?task_id=${this.taskId}`;
    
    try {
      this.ws = new WebSocket(url);
      
      this.ws.onopen = () => {
        // Check if we were disconnected while connecting
        if (this.isDisconnected) {
          this.ws.close();
          return;
        }
        this.hasConnected = true;
        console.log(`WebSocket connected for task ${this.taskId}`);
        this.reconnectAttempts = 0;
        this.isConnecting = false;
      };
      
      this.ws.onmessage = (event) => {
        if (this.isDisconnected) return;
        try {
          const data = JSON.parse(event.data);
          // Ignore ping messages
          if (data.type === 'ping') {
            return;
          }
          this.onMessage(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
      this.ws.onerror = (error) => {
        if (this.isDisconnected) return;
        // Only log error if we had successfully connected before
        if (this.hasConnected) {
          console.error('WebSocket error:', error);
        }
        this.isConnecting = false;
        if (this.onError && this.hasConnected) {
          this.onError(error);
        }
      };
      
      this.ws.onclose = (event) => {
        this.isConnecting = false;
        
        // Don't log or reconnect if intentionally disconnected
        if (this.isDisconnected) {
          return;
        }
        
        // Only log if we had successfully connected
        if (this.hasConnected) {
          console.log('WebSocket closed', event.code, event.reason);
          if (this.onClose) {
            this.onClose();
          }
        }
        
        // Only attempt reconnection if we had connected successfully before
        if (this.shouldReconnect && this.hasConnected && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnectAttempts++;
          const delay = Math.min(2000 * this.reconnectAttempts, 10000);
          setTimeout(() => {
            if (!this.isDisconnected) {
              console.log(`Reconnecting... (attempt ${this.reconnectAttempts})`);
              this.connect();
            }
          }, delay);
        }
      };
    } catch (error) {
      console.error('Error creating WebSocket:', error);
      this.isConnecting = false;
      if (this.onError) {
        this.onError(error);
      }
    }
  }

  disconnect() {
    this.shouldReconnect = false;
    this.isDisconnected = true;
    
    if (this.ws) {
      // Only close if not already closed/closing
      if (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING) {
        this.ws.close();
      }
      this.ws = null;
    }
  }

  send(message) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(message);
    }
  }
}

export default LogWebSocket;
