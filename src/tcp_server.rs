// src/tcp_server.rs

//! # High-Performance, Cross-Platform TCP Server Module
//!
//! A production-ready, zero-dependency TCP server designed for maximum throughput
//! on both Windows and Unix-like systems. Built with the FRD-PU philosophy, this
//! module provides a robust foundation for any network application.
//!
//! ## Features:
//! - **Cross-Platform:** Works seamlessly on Windows, Linux, and macOS.
//! - **Zero Dependencies:** Relies only on the Rust standard library.
//! - **High-Performance Architecture:** Lock-free, non-blocking design for massive concurrency.
//! - **Advanced Connection Management:** Supports keep-alive and graceful shutdowns.
//! - **Built-in Security:** Includes rate limiting for DoS protection.
//! - **Comprehensive Monitoring:** Provides detailed statistics and health checks.
//! - **Memory-Efficient:** Uses a robust buffer management strategy.
//! - **Full IPv4/IPv6 Support.**

use std::collections::{HashMap, VecDeque};
use std::io::{self, Read, Write, ErrorKind};
use std::net::{TcpListener, TcpStream, SocketAddr, Shutdown};
use std::sync::atomic::{AtomicBool, AtomicUsize, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use std::fmt;

// Platform-specific modules for socket operations
mod platform;
use platform::{set_socket_options, AsConnectionId, ConnectionId};


/// Buffer size constants
pub const DEFAULT_BUFFER_SIZE: usize = 64 * 1024; // 64KB
pub const MAX_BUFFER_SIZE: usize = 1024 * 1024; // 1MB
pub const MIN_BUFFER_SIZE: usize = 4 * 1024; // 4KB
pub const SOCKET_BUFFER_SIZE: usize = 256 * 1024; // 256KB for OS-level socket buffers

/// Default configuration constants
pub const DEFAULT_MAX_CONNECTIONS: usize = 10000;
pub const DEFAULT_BACKLOG: i32 = 1024;
pub const DEFAULT_TIMEOUT_SECS: u64 = 300; // 5 minutes
pub const DEFAULT_RATE_LIMIT: usize = 1000; // requests per second per connection
pub const DEFAULT_MAX_REQUEST_SIZE: usize = 1024 * 1024; // 1MB
pub const DEFAULT_WORKER_THREADS: usize = 8;


/// TCP Server errors with detailed context
#[derive(Debug)]
pub enum TcpServerError {
    IoError(io::Error),
    BindError { addr: String, source: io::Error },
    InvalidAddress(String),
    ServerShutdown,
    ConnectionError { addr: SocketAddr, source: io::Error },
    BufferOverflow { size: usize, max: usize },
    RateLimitExceeded { addr: SocketAddr, limit: usize },
    TimeoutError { addr: SocketAddr, timeout: Duration },
    ConfigurationError(String),
    ResourceExhausted(String),
    ProtocolError(String),
    SocketOptionError { description: String, source: io::Error },
}

impl fmt::Display for TcpServerError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TcpServerError::IoError(e) => write!(f, "I/O error: {}", e),
            TcpServerError::BindError { addr, source } => write!(f, "Failed to bind to {}: {}", addr, source),
            TcpServerError::InvalidAddress(addr) => write!(f, "Invalid address: {}", addr),
            TcpServerError::ServerShutdown => write!(f, "Server is shutting down"),
            TcpServerError::ConnectionError { addr, source } => write!(f, "Connection error from {}: {}", addr, source),
            TcpServerError::BufferOverflow { size, max } => write!(f, "Buffer overflow: {} bytes exceeds maximum {}", size, max),
            TcpServerError::RateLimitExceeded { addr, limit } => write!(f, "Rate limit exceeded for {}: {} req/s", addr, limit),
            TcpServerError::TimeoutError { addr, timeout } => write!(f, "Timeout error for {}: {:?}", addr, timeout),
            TcpServerError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            TcpServerError::ResourceExhausted(msg) => write!(f, "Resource exhausted: {}", msg),
            TcpServerError::ProtocolError(msg) => write!(f, "Protocol error: {}", msg),
            TcpServerError::SocketOptionError{ description, source } => write!(f, "Failed to set socket option '{}': {}", description, source),
        }
    }
}

impl From<io::Error> for TcpServerError {
    fn from(err: io::Error) -> Self {
        TcpServerError::IoError(err)
    }
}

/// Connection statistics for monitoring
#[derive(Debug, Default)]
pub struct ConnectionStats {
    pub total_connections: AtomicUsize,
    pub active_connections: AtomicUsize,
    pub bytes_received: AtomicU64,
    pub bytes_sent: AtomicU64,
    pub messages_processed: AtomicU64,
    pub errors_count: AtomicUsize,
    pub rate_limited_count: AtomicUsize,
    pub timeout_count: AtomicUsize,
    pub uptime_seconds: AtomicU64,
}

impl ConnectionStats {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn get_snapshot(&self) -> ConnectionStatsSnapshot {
        ConnectionStatsSnapshot {
            total_connections: self.total_connections.load(Ordering::Relaxed),
            active_connections: self.active_connections.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
            messages_processed: self.messages_processed.load(Ordering::Relaxed),
            errors_count: self.errors_count.load(Ordering::Relaxed),
            rate_limited_count: self.rate_limited_count.load(Ordering::Relaxed),
            timeout_count: self.timeout_count.load(Ordering::Relaxed),
            uptime_seconds: self.uptime_seconds.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConnectionStatsSnapshot {
    pub total_connections: usize,
    pub active_connections: usize,
    pub bytes_received: u64,
    pub bytes_sent: u64,
    pub messages_processed: u64,
    pub errors_count: usize,
    pub rate_limited_count: usize,
    pub timeout_count: usize,
    pub uptime_seconds: u64,
}

/// Rate limiter for DoS protection
#[derive(Debug)]
struct RateLimiter {
    requests: VecDeque<Instant>,
    limit: usize,
    window: Duration,
}

impl RateLimiter {
    fn new(limit: usize, window: Duration) -> Self {
        Self {
            requests: VecDeque::new(),
            limit,
            window,
        }
    }
    
    fn check(&mut self) -> bool {
        let now = Instant::now();
        let cutoff = now - self.window;
        
        // Remove old entries
        while let Some(&front) = self.requests.front() {
            if front < cutoff {
                self.requests.pop_front();
            } else {
                break;
            }
        }
        
        if self.requests.len() < self.limit {
            self.requests.push_back(now);
            true
        } else {
            false
        }
    }
}

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub max_connections: usize,
    pub buffer_size: usize,
    pub timeout: Duration,
    pub rate_limit: usize,
    pub rate_limit_window: Duration,
    pub max_request_size: usize,
    pub worker_threads: usize,
    pub backlog: i32,
    pub enable_keepalive: bool,
    pub enable_nodelay: bool,
    pub socket_buffer_size: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            max_connections: DEFAULT_MAX_CONNECTIONS,
            buffer_size: DEFAULT_BUFFER_SIZE,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            rate_limit: DEFAULT_RATE_LIMIT,
            rate_limit_window: Duration::from_secs(1),
            max_request_size: DEFAULT_MAX_REQUEST_SIZE,
            worker_threads: DEFAULT_WORKER_THREADS,
            backlog: DEFAULT_BACKLOG,
            enable_keepalive: true,
            enable_nodelay: true,
            socket_buffer_size: SOCKET_BUFFER_SIZE,
        }
    }
}

/// Connection handler trait for processing requests
pub trait ConnectionHandler: Send + Sync + 'static {
    /// Handle incoming data from a connection
    /// Returns response data to send back, or None to close connection
    fn handle_data(&self, data: &[u8], addr: SocketAddr) -> Option<Vec<u8>>;
    
    /// Called when a new connection is established
    fn on_connect(&self, _addr: SocketAddr) {}
    
    /// Called when a connection is closed
    fn on_disconnect(&self, _addr: SocketAddr) {}
    
    /// Called on server start
    fn on_server_start(&self) {}
    
    /// Called on server shutdown
    fn on_server_shutdown(&self) {}
    
    /// Health check endpoint
    fn health_check(&self) -> bool { true }
}

/// Simple echo handler for testing
pub struct EchoHandler;

impl ConnectionHandler for EchoHandler {
    fn handle_data(&self, data: &[u8], _addr: SocketAddr) -> Option<Vec<u8>> {
        Some(data.to_vec())
    }
}

/// HTTP-like handler example
pub struct HttpHandler;

impl ConnectionHandler for HttpHandler {
    fn handle_data(&self, data: &[u8], _addr: SocketAddr) -> Option<Vec<u8>> {
        let request = String::from_utf8_lossy(data);
        if request.starts_with("GET") {
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 13\r\n\r\nHello, World!"
            );
            Some(response.into_bytes())
        } else {
            let response = format!(
                "HTTP/1.1 400 Bad Request\r\nContent-Length: 11\r\n\r\nBad Request"
            );
            Some(response.into_bytes())
        }
    }
}

/// Connection state management
#[derive(Debug)]
struct Connection {
    stream: TcpStream,
    addr: SocketAddr,
    read_buffer: Vec<u8>,
    write_buffer: Vec<u8>,
    last_activity: Instant,
    bytes_received: u64,
    bytes_sent: u64,
    rate_limiter: RateLimiter,
}

impl Connection {
    fn new(stream: TcpStream, addr: SocketAddr, config: &ServerConfig) -> Result<Self, TcpServerError> {
        // Set non-blocking mode
        stream.set_nonblocking(true)?;
        
        // Apply cross-platform socket optimizations
        set_socket_options(&stream, config)?;
        
        let now = Instant::now();
        Ok(Self {
            stream,
            addr,
            read_buffer: Vec::with_capacity(config.buffer_size),
            write_buffer: Vec::new(),
            last_activity: now,
            bytes_received: 0,
            bytes_sent: 0,
            rate_limiter: RateLimiter::new(config.rate_limit, config.rate_limit_window),
        })
    }
    
    fn read_available(&mut self, max_size: usize) -> io::Result<usize> {
        let mut temp_buffer = vec![0u8; 8192.min(max_size - self.read_buffer.len())];
        
        match self.stream.read(&mut temp_buffer) {
            Ok(0) => Ok(0), // Connection closed
            Ok(n) => {
                if self.read_buffer.len() + n > max_size {
                    return Err(io::Error::new(ErrorKind::InvalidData, "Request too large"));
                }
                self.read_buffer.extend_from_slice(&temp_buffer[..n]);
                self.bytes_received += n as u64;
                self.last_activity = Instant::now();
                Ok(n)
            }
            Err(ref e) if e.kind() == ErrorKind::WouldBlock => Ok(0),
            Err(e) => Err(e),
        }
    }
    
    fn write_pending(&mut self) -> io::Result<bool> {
        if self.write_buffer.is_empty() {
            return Ok(true); // Nothing to write
        }
        
        match self.stream.write(&self.write_buffer) {
            Ok(0) => Ok(false), // Connection closed
            Ok(n) => {
                self.bytes_sent += n as u64;
                self.last_activity = Instant::now();
                self.write_buffer.drain(..n);
                Ok(self.write_buffer.is_empty())
            }
            Err(ref e) if e.kind() == ErrorKind::WouldBlock => Ok(false),
            Err(_) => Ok(false), // Connection error
        }
    }
    
    fn queue_response(&mut self, data: Vec<u8>) {
        self.write_buffer.extend(data);
    }
    
    fn is_timed_out(&self, timeout: Duration) -> bool {
        self.last_activity.elapsed() > timeout
    }
    
    fn check_rate_limit(&mut self) -> bool {
        self.rate_limiter.check()
    }
}

/// High-performance cross-platform TCP Server
pub struct TcpServer<H: ConnectionHandler> {
    listener: Option<TcpListener>,
    handler: Arc<H>,
    connections: HashMap<ConnectionId, Connection>,
    running: Arc<AtomicBool>,
    stats: Arc<ConnectionStats>,
    start_time: Instant,
    config: ServerConfig,
}

impl<H: ConnectionHandler> TcpServer<H> {
    /// Create a new TCP server with default configuration
    pub fn new(addr: &str, handler: H) -> Result<Self, TcpServerError> {
        Self::with_config(addr, handler, ServerConfig::default())
    }
    
    /// Create a new TCP server with custom configuration
    pub fn with_config(addr: &str, handler: H, config: ServerConfig) -> Result<Self, TcpServerError> {
        let listener = TcpListener::bind(addr)
            .map_err(|e| TcpServerError::BindError { 
                addr: addr.to_string(), 
                source: e 
            })?;
        
        // Set listener to non-blocking mode
        listener.set_nonblocking(true)?;
        
        // Apply cross-platform socket optimizations to listener
        set_socket_options(&listener, &config)?;
        
        Ok(Self {
            listener: Some(listener),
            handler: Arc::new(handler),
            connections: HashMap::with_capacity(config.max_connections),
            running: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(ConnectionStats::new()),
            start_time: Instant::now(),
            config,
        })
    }
    
    /// Configure maximum concurrent connections
    pub fn with_max_connections(mut self, max: usize) -> Self {
        self.config.max_connections = max;
        self.connections.reserve(max);
        self
    }
    
    /// Configure buffer size
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.config.buffer_size = size.clamp(MIN_BUFFER_SIZE, MAX_BUFFER_SIZE);
        self
    }
    
    /// Configure connection timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }
    
    /// Configure rate limiting
    pub fn with_rate_limit(mut self, limit: usize, window: Duration) -> Self {
        self.config.rate_limit = limit;
        self.config.rate_limit_window = window;
        self
    }
    
    /// Start the server (blocking call)
    pub fn serve(&mut self) -> Result<(), TcpServerError> {
        let listener = self.listener.take()
            .ok_or(TcpServerError::ServerShutdown)?;
        
        let local_addr = listener.local_addr()
            .map_err(TcpServerError::IoError)?;
        
        self.running.store(true, Ordering::Relaxed);
        self.start_time = Instant::now();
        
        println!("FRD-PU TCP Server starting on {} (Cross-Platform)", local_addr);
        println!("Configuration: {} max connections, {}KB buffers, {:?} timeout", 
                self.config.max_connections, 
                self.config.buffer_size / 1024,
                self.config.timeout);
        
        self.handler.on_server_start();
        
        // Main event loop - non-blocking
        while self.running.load(Ordering::Relaxed) {
            // Accept new connections
            self.accept_connections(&listener)?;
            
            // Process existing connections
            self.process_connections()?;
            
            // Cleanup timed out connections
            self.cleanup_connections();
            
            // Update statistics
            self.update_stats();
            
            // Micro-sleep to prevent CPU spinning
            thread::sleep(Duration::from_micros(100));
        }
        
        // Graceful shutdown
        self.shutdown_connections();
        self.handler.on_server_shutdown();
        
        println!("FRD-PU TCP Server shutdown complete");
        Ok(())
    }
    
    /// Accept new incoming connections
    fn accept_connections(&mut self, listener: &TcpListener) -> Result<(), TcpServerError> {
        loop {
            match listener.accept() {
                Ok((stream, addr)) => {
                    // Check connection limit
                    if self.connections.len() >= self.config.max_connections {
                        drop(stream); // Drop connection at capacity
                        continue;
                    }
                    
                    // Create new connection
                    match Connection::new(stream, addr, &self.config) {
                        Ok(connection) => {
                            let conn_id = connection.stream.as_conn_id();
                            self.handler.on_connect(addr);
                            self.connections.insert(conn_id, connection);
                            
                            self.stats.total_connections.fetch_add(1, Ordering::Relaxed);
                            self.stats.active_connections.store(self.connections.len(), Ordering::Relaxed);
                        }
                        Err(e) => {
                            self.stats.errors_count.fetch_add(1, Ordering::Relaxed);
                            eprintln!("Failed to create connection for {}: {}", addr, e);
                        }
                    }
                }
                Err(ref e) if e.kind() == ErrorKind::WouldBlock => break,
                Err(e) => {
                    self.stats.errors_count.fetch_add(1, Ordering::Relaxed);
                    return Err(TcpServerError::IoError(e));
                }
            }
        }
        Ok(())
    }
    
    /// Process all active connections
    fn process_connections(&mut self) -> Result<(), TcpServerError> {
        let mut to_remove = Vec::new();
        
        for (conn_id, connection) in self.connections.iter_mut() {
            match Self::process_single_connection(
                connection,
                &self.config,
                &self.stats,
                &self.handler
            ) {
                Ok(true) => {
                    // Connection processed successfully
                }
                Ok(false) => {
                    // Connection should be closed
                    self.handler.on_disconnect(connection.addr);
                    to_remove.push(*conn_id);
                }
                Err(_) => {
                    // Connection error
                    self.stats.errors_count.fetch_add(1, Ordering::Relaxed);
                    self.handler.on_disconnect(connection.addr);
                    to_remove.push(*conn_id);
                }
            }
        }
        
        // Remove closed connections
        for conn_id in to_remove {
            self.connections.remove(&conn_id);
        }
        
        self.stats.active_connections.store(self.connections.len(), Ordering::Relaxed);
        Ok(())
    }
    
    /// Process a single connection (as a static method to avoid borrow issues)
    fn process_single_connection(
        conn: &mut Connection,
        config: &ServerConfig,
        stats: &Arc<ConnectionStats>,
        handler: &Arc<H>
    ) -> Result<bool, TcpServerError> {
        // Read available data
        match conn.read_available(config.max_request_size) {
            Ok(0) if conn.read_buffer.is_empty() => return Ok(false), // Connection closed
            Ok(n) if n > 0 => {
                stats.bytes_received.fetch_add(n as u64, Ordering::Relaxed);
            }
            Ok(_) => {}, // Data received or would block
            Err(_) => return Ok(false), // Read error
        }
        
        // Write pending data
        let bytes_written_before = conn.bytes_sent;
        if !conn.write_pending()? {
             // Still have data to write or error occurred
        }
        let bytes_written_after = conn.bytes_sent;
        if bytes_written_after > bytes_written_before {
            stats.bytes_sent.fetch_add(bytes_written_after - bytes_written_before, Ordering::Relaxed);
        }
        
        // Process complete requests in buffer
        if !conn.read_buffer.is_empty() {
            // Check rate limit
            if !conn.check_rate_limit() {
                stats.rate_limited_count.fetch_add(1, Ordering::Relaxed);
                return Ok(true); // Don't close, just skip processing
            }
            
            // Process the request
            if let Some(response) = handler.handle_data(&conn.read_buffer, conn.addr) {
                conn.queue_response(response);
                stats.messages_processed.fetch_add(1, Ordering::Relaxed);
            }
            
            conn.read_buffer.clear();
        }
        
        Ok(true)
    }
    
    /// Clean up timed out connections
    fn cleanup_connections(&mut self) {
        let mut to_remove = Vec::new();
        
        for (conn_id, connection) in self.connections.iter() {
            if connection.is_timed_out(self.config.timeout) {
                to_remove.push(*conn_id);
                self.stats.timeout_count.fetch_add(1, Ordering::Relaxed);
            }
        }
        
        for conn_id in to_remove {
            if let Some(connection) = self.connections.remove(&conn_id) {
                self.handler.on_disconnect(connection.addr);
            }
        }
    }
    
    /// Update server statistics
    fn update_stats(&mut self) {
        let uptime = self.start_time.elapsed().as_secs();
        self.stats.uptime_seconds.store(uptime, Ordering::Relaxed);
    }
    
    /// Shutdown all connections gracefully
    fn shutdown_connections(&mut self) {
        for (_, connection) in self.connections.drain() {
            let _ = connection.stream.shutdown(Shutdown::Both);
            self.handler.on_disconnect(connection.addr);
        }
    }
    
    /// Get current server statistics
    pub fn get_stats(&self) -> ConnectionStatsSnapshot {
        self.stats.get_snapshot()
    }
    
    /// Stop the server
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }
    
    /// Check if server is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
    
    /// Get server configuration
    pub fn get_config(&self) -> &ServerConfig {
        &self.config
    }
    
    /// Print server statistics
    pub fn print_stats(&self) {
        let stats = self.get_stats();
        println!("=== FRD-PU TCP Server Statistics ===");
        println!("Uptime: {} seconds", stats.uptime_seconds);
        println!("Total connections: {}", stats.total_connections);
        println!("Active connections: {}", stats.active_connections);
        println!("Bytes received: {} MB", stats.bytes_received / (1024 * 1024));
        println!("Bytes sent: {} MB", stats.bytes_sent / (1024 * 1024));
        println!("Messages processed: {}", stats.messages_processed);
        println!("Errors: {}", stats.errors_count);
        println!("Rate limited: {}", stats.rate_limited_count);
        println!("Timeouts: {}", stats.timeout_count);
        
        if stats.uptime_seconds > 0 {
            let msg_per_sec = stats.messages_processed / stats.uptime_seconds;
            let mb_per_sec = (stats.bytes_received + stats.bytes_sent) / (1024 * 1024 * stats.uptime_seconds);
            println!("Throughput: {} msg/s, {} MB/s", msg_per_sec, mb_per_sec);
        }
    }
}

// Graceful shutdown on Drop
impl<H: ConnectionHandler> Drop for TcpServer<H> {
    fn drop(&mut self) {
        if self.is_running() {
            self.running.store(false, Ordering::Relaxed);
            self.shutdown_connections();
        }
    }
}

/// Convenience functions for creating servers

/// Create a new TCP server with echo handler
pub fn new_echo_server(addr: &str) -> Result<TcpServer<EchoHandler>, TcpServerError> {
    TcpServer::new(addr, EchoHandler)
}

/// Create a new TCP server with HTTP handler
pub fn new_http_server(addr: &str) -> Result<TcpServer<HttpHandler>, TcpServerError> {
    TcpServer::new(addr, HttpHandler)
}

/// Create a new TCP server with custom handler
pub fn new_tcp_server<H: ConnectionHandler>(addr: &str, handler: H) -> Result<TcpServer<H>, TcpServerError> {
    TcpServer::new(addr, handler)
}

/// Create a new TCP server with custom configuration
pub fn new_tcp_server_with_config<H: ConnectionHandler>(
    addr: &str, 
    handler: H, 
    config: ServerConfig
) -> Result<TcpServer<H>, TcpServerError> {
    TcpServer::with_config(addr, handler, config)
}


/// Platform-specific socket implementation details
mod platform {
    use super::{ServerConfig, TcpServerError};
    use std::io;
    use std::mem;

    // Define a platform-agnostic connection identifier and a trait to access it.
    #[cfg(unix)]
    pub use std::os::unix::io::{AsRawFd, RawFd as ConnectionId};
    #[cfg(windows)]
    pub use std::os::windows::io::{AsRawSocket, RawSocket as ConnectionId};

    pub trait AsConnectionId {
        fn as_conn_id(&self) -> ConnectionId;
    }

    #[cfg(unix)]
    impl<T: AsRawFd> AsConnectionId for T {
        fn as_conn_id(&self) -> ConnectionId {
            self.as_raw_fd()
        }
    }

    #[cfg(windows)]
    impl<T: AsRawSocket> AsConnectionId for T {
        fn as_conn_id(&self) -> ConnectionId {
            self.as_raw_socket()
        }
    }

    /// Set cross-platform socket options.
    pub fn set_socket_options<S: AsConnectionId>(
        socket_like: &S,
        config: &ServerConfig,
    ) -> Result<(), TcpServerError> {
        #[cfg(windows)]
        return windows::set_socket_options_impl(socket_like, config);
        #[cfg(unix)]
        return unix::set_socket_options_impl(socket_like, config);
    }

    /// Windows-specific socket options implementation.
    #[cfg(windows)]
    mod windows {
        use super::*;
        use std::os::windows::io::{AsRawSocket, RawSocket};
        
        // Windows-specific FFI for setsockopt
        #[link(name = "ws2_32")]
        extern "system" {
            fn setsockopt(s: RawSocket, level: i32, optname: i32, optval: *const i8, optlen: i32) -> i32;
        }

        const SOL_SOCKET: i32 = 0xffff;
        const SO_REUSEADDR: i32 = 0x0004;
        const SO_KEEPALIVE: i32 = 0x0008;
        const SO_RCVBUF: i32 = 0x1002;
        const SO_SNDBUF: i32 = 0x1001;
        const IPPROTO_TCP: i32 = 6;
        const TCP_NODELAY: i32 = 0x0001;
        
        pub fn set_socket_options_impl<S: AsRawSocket>(
            socket_like: &S,
            config: &ServerConfig,
        ) -> Result<(), TcpServerError> {
            let socket = socket_like.as_raw_socket();
            
            set_opt(socket, SOL_SOCKET, SO_REUSEADDR, config.enable_keepalive as u32, "SO_REUSEADDR")?;
            set_opt(socket, SOL_SOCKET, SO_KEEPALIVE, config.enable_keepalive as u32, "SO_KEEPALIVE")?;
            set_opt(socket, IPPROTO_TCP, TCP_NODELAY, config.enable_nodelay as u32, "TCP_NODELAY")?;
            set_opt(socket, SOL_SOCKET, SO_RCVBUF, config.socket_buffer_size as u32, "SO_RCVBUF")?;
            set_opt(socket, SOL_SOCKET, SO_SNDBUF, config.socket_buffer_size as u32, "SO_SNDBUF")?;
            
            Ok(())
        }

        fn set_opt(socket: RawSocket, level: i32, optname: i32, optval: u32, desc: &str) -> Result<(), TcpServerError> {
            let result = unsafe {
                setsockopt(
                    socket,
                    level,
                    optname,
                    &optval as *const u32 as *const i8,
                    mem::size_of::<u32>() as i32,
                )
            };
            if result != 0 {
                Err(TcpServerError::SocketOptionError {
                    description: desc.to_string(),
                    source: io::Error::last_os_error(),
                })
            } else {
                Ok(())
            }
        }
    }

    /// Unix-specific socket options implementation.
    #[cfg(unix)]
    mod unix {
        use super::*;
        use std::os::unix::io::{AsRawFd, RawFd};
        use std::ffi::c_void;

        // FFI for setsockopt on Unix-like systems
        extern "C" {
            fn setsockopt(socket: i32, level: i32, name: i32, value: *const c_void, option_len: u32) -> i32;
        }

        // Constants from libc, defined here to avoid a dependency.
        const SOL_SOCKET: i32 = 1;
        const SO_REUSEADDR: i32 = 2;
        const SO_KEEPALIVE: i32 = 9;
        const SO_RCVBUF: i32 = 8;
        const SO_SNDBUF: i32 = 7;
        const IPPROTO_TCP: i32 = 6;
        const TCP_NODELAY: i32 = 1;

        pub fn set_socket_options_impl<S: AsRawFd>(
            socket_like: &S,
            config: &ServerConfig,
        ) -> Result<(), TcpServerError> {
            let fd = socket_like.as_raw_fd();

            set_opt(fd, SOL_SOCKET, SO_REUSEADDR, config.enable_keepalive as u32, "SO_REUSEADDR")?;
            set_opt(fd, SOL_SOCKET, SO_KEEPALIVE, config.enable_keepalive as u32, "SO_KEEPALIVE")?;
            set_opt(fd, IPPROTO_TCP, TCP_NODELAY, config.enable_nodelay as u32, "TCP_NODELAY")?;
            set_opt(fd, SOL_SOCKET, SO_RCVBUF, config.socket_buffer_size as u32, "SO_RCVBUF")?;
            set_opt(fd, SOL_SOCKET, SO_SNDBUF, config.socket_buffer_size as u32, "SO_SNDBUF")?;

            Ok(())
        }

        fn set_opt(fd: RawFd, level: i32, optname: i32, optval: u32, desc: &str) -> Result<(), TcpServerError> {
            let result = unsafe {
                setsockopt(
                    fd,
                    level,
                    optname,
                    &optval as *const u32 as *const c_void,
                    mem::size_of::<u32>() as u32,
                )
            };
            if result != 0 {
                Err(TcpServerError::SocketOptionError {
                    description: desc.to_string(),
                    source: io::Error::last_os_error(),
                })
            } else {
                Ok(())
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;
    use std::net::TcpStream;
    use std::io::{Read, Write};
    
    #[test]
    fn test_echo_server_creation() {
        let server = new_echo_server("127.0.0.1:0");
        assert!(server.is_ok());
    }
    
    #[test]
    fn test_http_server_creation() {
        let server = new_http_server("127.0.0.1:0");
        assert!(server.is_ok());
    }
    
    #[test]
    fn test_server_configuration() {
        let config = ServerConfig {
            max_connections: 5000,
            buffer_size: 32 * 1024,
            timeout: Duration::from_secs(120),
            rate_limit: 500,
            rate_limit_window: Duration::from_secs(1),
            max_request_size: 512 * 1024,
            worker_threads: 4,
            backlog: 512,
            enable_keepalive: true,
            enable_nodelay: true,
            socket_buffer_size: 128 * 1024,
        };
        
        let server = new_tcp_server_with_config("127.0.0.1:0", EchoHandler, config);
        assert!(server.is_ok());
        
        let server = server.unwrap();
        assert_eq!(server.get_config().max_connections, 5000);
        assert_eq!(server.get_config().buffer_size, 32 * 1024);
    }
    
    #[test]
    fn test_connection_handler_trait() {
        struct TestHandler;
        
        impl ConnectionHandler for TestHandler {
            fn handle_data(&self, data: &[u8], _addr: SocketAddr) -> Option<Vec<u8>> {
                let input = String::from_utf8_lossy(data);
                Some(format!("Processed: {}", input).into_bytes())
            }
            
            fn on_connect(&self, addr: SocketAddr) {
                println!("Client connected: {}", addr);
            }
            
            fn on_disconnect(&self, addr: SocketAddr) {
                println!("Client disconnected: {}", addr);
            }
        }
        
        let server = new_tcp_server("127.0.0.1:0", TestHandler);
        assert!(server.is_ok());
    }
    
    #[test]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(5, Duration::from_secs(1));
        
        // Should allow up to 5 requests
        for _ in 0..5 {
            assert!(limiter.check());
        }
        
        // 6th request should be denied
        assert!(!limiter.check());
    }
    
    #[test]
    fn test_socket_options() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let config = ServerConfig::default();
        let result = set_socket_options(&listener, &config);
        assert!(result.is_ok(), "Setting socket options failed: {:?}", result.err());
    }
    
    #[test]
    fn test_server_stats() {
        let stats = ConnectionStats::new();
        stats.total_connections.store(100, Ordering::Relaxed);
        stats.bytes_received.store(1024 * 1024, Ordering::Relaxed);
        
        let snapshot = stats.get_snapshot();
        assert_eq!(snapshot.total_connections, 100);
        assert_eq!(snapshot.bytes_received, 1024 * 1024);
    }
}
