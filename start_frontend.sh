#!/bin/bash
# Navigate to Angular project directory
cd "$(dirname "$0")/frontend" || {
    echo "❌ Could not find ocean-search-ui directory"
    exit 1
}

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Start Angular server
echo "🚀 Starting Angular frontend..."
ng serve --host 0.0.0.0 --allowed-hosts
