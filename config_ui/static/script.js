// LogGuard ML Configuration UI JavaScript

// Global variables
let notifications = [];

// Utility functions
function showNotification(message, type = 'info', duration = 5000) {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    // Add close button
    const closeBtn = document.createElement('button');
    closeBtn.innerHTML = '×';
    closeBtn.style.cssText = `
        background: none;
        border: none;
        color: white;
        font-size: 1.2rem;
        cursor: pointer;
        margin-left: 1rem;
        opacity: 0.8;
    `;
    closeBtn.onclick = () => notification.remove();
    notification.appendChild(closeBtn);
    
    document.body.appendChild(notification);
    
    // Auto remove after duration
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, duration);
    
    notifications.push(notification);
}

function clearNotifications() {
    notifications.forEach(notification => {
        if (notification.parentNode) {
            notification.remove();
        }
    });
    notifications = [];
}

// API helper functions
async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(endpoint, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}

// Dashboard functionality
function initializeDashboard() {
    if (!document.querySelector('.dashboard-grid')) return;
    
    // Load system status
    loadSystemStatus();
    loadPerformanceMetrics();
    loadRecentLogs();
    
    // Set up auto-refresh
    setInterval(loadSystemStatus, 30000); // Every 30 seconds
    setInterval(loadPerformanceMetrics, 60000); // Every minute
}

async function loadSystemStatus() {
    try {
        const status = await apiCall('/api/status');
        updateSystemStatus(status);
    } catch (error) {
        console.error('Failed to load system status:', error);
    }
}

function updateSystemStatus(status) {
    // Update status indicators
    const statusElement = document.getElementById('system-status');
    if (statusElement) {
        statusElement.className = `status ${status.overall}`;
        statusElement.textContent = status.overall.toUpperCase();
    }
    
    // Update component statuses
    Object.entries(status.components || {}).forEach(([component, componentStatus]) => {
        const element = document.getElementById(`${component}-status`);
        if (element) {
            element.className = `status ${componentStatus}`;
            element.textContent = componentStatus.toUpperCase();
        }
    });
}

async function loadPerformanceMetrics() {
    try {
        const metrics = await apiCall('/api/metrics');
        updatePerformanceMetrics(metrics);
    } catch (error) {
        console.error('Failed to load performance metrics:', error);
    }
}

function updatePerformanceMetrics(metrics) {
    // Update metric values
    Object.entries(metrics).forEach(([metric, value]) => {
        const element = document.getElementById(`metric-${metric}`);
        if (element) {
            element.textContent = typeof value === 'number' ? 
                value.toLocaleString() : value;
        }
    });
}

async function loadRecentLogs() {
    try {
        const logs = await apiCall('/api/logs/recent');
        updateRecentLogs(logs);
    } catch (error) {
        console.error('Failed to load recent logs:', error);
    }
}

function updateRecentLogs(logs) {
    const container = document.getElementById('recent-logs');
    if (!container) return;
    
    container.innerHTML = '';
    
    logs.slice(0, 10).forEach(log => {
        const logElement = document.createElement('div');
        logElement.className = 'log-entry';
        logElement.innerHTML = `
            <div class="log-time">${new Date(log.timestamp).toLocaleTimeString()}</div>
            <div class="log-level ${log.level.toLowerCase()}">${log.level}</div>
            <div class="log-message">${log.message}</div>
        `;
        container.appendChild(logElement);
    });
}

// Configuration functionality
function initializeConfiguration() {
    if (!document.getElementById('config-form')) return;
    
    // Load current configuration
    loadConfiguration();
    
    // Set up form validation
    setupConfigurationValidation();
    
    // Set up preview updates
    setupConfigurationPreview();
}

async function loadConfiguration() {
    try {
        const config = await apiCall('/api/config');
        populateConfigurationForm(config);
    } catch (error) {
        console.error('Failed to load configuration:', error);
        showNotification('Failed to load configuration', 'error');
    }
}

function populateConfigurationForm(config) {
    // Populate form fields with configuration values
    Object.entries(config).forEach(([section, settings]) => {
        if (typeof settings === 'object' && settings !== null) {
            Object.entries(settings).forEach(([key, value]) => {
                const input = document.querySelector(`[name="${section}.${key}"]`) || 
                            document.querySelector(`[name="${key}"]`);
                if (input) {
                    if (input.type === 'checkbox') {
                        input.checked = Boolean(value);
                    } else {
                        input.value = value;
                    }
                }
            });
        }
    });
}

function setupConfigurationValidation() {
    const form = document.getElementById('config-form');
    if (!form) return;
    
    // Add real-time validation
    form.querySelectorAll('input, select').forEach(input => {
        input.addEventListener('change', validateField);
        input.addEventListener('blur', validateField);
    });
}

function validateField(event) {
    const field = event.target;
    const value = field.type === 'checkbox' ? field.checked : field.value;
    
    // Remove existing validation classes
    field.classList.remove('valid', 'invalid');
    
    // Basic validation
    let isValid = true;
    
    if (field.required && !value) {
        isValid = false;
    }
    
    if (field.type === 'number') {
        const num = parseFloat(value);
        if (isNaN(num) || 
            (field.min && num < parseFloat(field.min)) ||
            (field.max && num > parseFloat(field.max))) {
            isValid = false;
        }
    }
    
    // Add validation class
    field.classList.add(isValid ? 'valid' : 'invalid');
    
    return isValid;
}

function setupConfigurationPreview() {
    const form = document.getElementById('config-form');
    const preview = document.getElementById('config-preview-content');
    
    if (!form || !preview) return;
    
    // Update preview when form changes
    form.addEventListener('change', updateConfigurationPreview);
    form.addEventListener('input', debounce(updateConfigurationPreview, 500));
}

function updateConfigurationPreview() {
    const form = document.getElementById('config-form');
    const preview = document.getElementById('config-preview-content');
    
    if (!form || !preview) return;
    
    const formData = new FormData(form);
    const config = {};
    
    // Convert form data to configuration object
    formData.forEach((value, key) => {
        const parts = key.split('.');
        if (parts.length === 2) {
            const [section, setting] = parts;
            if (!config[section]) config[section] = {};
            config[section][setting] = value;
        } else {
            config[key] = value;
        }
    });
    
    // Display as formatted JSON
    preview.textContent = JSON.stringify(config, null, 2);
}

// Plugin management functionality
function initializePlugins() {
    if (!document.querySelector('.plugins-section')) return;
    
    // Load plugin data
    loadPlugins();
    
    // Set up plugin actions
    setupPluginActions();
}

async function loadPlugins() {
    try {
        const plugins = await apiCall('/api/plugins');
        updatePluginsDisplay(plugins);
    } catch (error) {
        console.error('Failed to load plugins:', error);
        showNotification('Failed to load plugins', 'error');
    }
}

function updatePluginsDisplay(plugins) {
    // Update plugin counts and statuses
    Object.entries(plugins).forEach(([type, pluginList]) => {
        const container = document.getElementById(type.replace('_', '-'));
        if (container) {
            updatePluginGrid(container, pluginList);
        }
    });
}

function updatePluginGrid(container, plugins) {
    const grid = container.querySelector('.plugins-grid');
    if (!grid) return;
    
    // Clear existing content
    grid.innerHTML = '';
    
    if (plugins.length === 0) {
        grid.innerHTML = '<div class="no-plugins">No plugins available</div>';
        return;
    }
    
    // Add plugin cards
    plugins.forEach(plugin => {
        const card = createPluginCard(plugin);
        grid.appendChild(card);
    });
}

function createPluginCard(plugin) {
    const card = document.createElement('div');
    card.className = 'plugin-card';
    card.innerHTML = `
        <div class="plugin-header">
            <h3>${plugin.name}</h3>
            <span class="plugin-status ${plugin.enabled ? 'enabled' : 'disabled'}">
                ${plugin.enabled ? 'Enabled' : 'Disabled'}
            </span>
        </div>
        <div class="plugin-body">
            <p class="plugin-description">${plugin.description}</p>
            <div class="plugin-details">
                <div class="detail-item"><strong>Version:</strong> ${plugin.version}</div>
                <div class="detail-item"><strong>Type:</strong> ${plugin.type}</div>
            </div>
        </div>
        <div class="plugin-actions">
            <button class="btn btn-sm ${plugin.enabled ? 'btn-warning' : 'btn-success'}"
                    onclick="togglePlugin('${plugin.name}', '${plugin.type}')">
                ${plugin.enabled ? 'Disable' : 'Enable'}
            </button>
            <button class="btn btn-sm btn-info"
                    onclick="testPlugin('${plugin.name}', '${plugin.type}')">
                Test
            </button>
        </div>
    `;
    return card;
}

function setupPluginActions() {
    // Plugin loading button
    const loadButton = document.getElementById('load-plugins');
    if (loadButton) {
        loadButton.addEventListener('click', async () => {
            try {
                const result = await apiCall('/api/plugins/reload', { method: 'POST' });
                showNotification(`Loaded ${result.count} plugins`, 'success');
                loadPlugins(); // Refresh display
            } catch (error) {
                showNotification('Failed to load plugins', 'error');
            }
        });
    }
    
    // Plugin refresh button
    const refreshButton = document.getElementById('refresh-plugins');
    if (refreshButton) {
        refreshButton.addEventListener('click', () => {
            loadPlugins();
        });
    }
}

// Utility function for debouncing
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Theme handling
function initializeTheme() {
    const savedTheme = localStorage.getItem('logguard-theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    
    // Theme toggle button
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('logguard-theme', newTheme);
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize common functionality
    initializeTheme();
    
    // Initialize page-specific functionality
    initializeDashboard();
    initializeConfiguration();
    initializePlugins();
    
    // Set up global error handling
    window.addEventListener('unhandledrejection', function(event) {
        console.error('Unhandled promise rejection:', event.reason);
        showNotification('An unexpected error occurred', 'error');
    });
    
    console.log('LogGuard ML Configuration UI initialized');
});
