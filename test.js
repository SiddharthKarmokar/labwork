// Import the WebSocket library
const WebSocket = require('ws');

// Now your original code will work
const ws = new WebSocket("ws://localhost:3000/ws?deploymentId=test-deployment");

// Optional: Add some basic event listeners to test the connection
ws.on('open', () => {
    console.log('Successfully connected to the WebSocket server!');
});

ws.on('error', (error) => {
    console.error('WebSocket Error:', error);
});