from flask import Flask, request, jsonify
from ethereum import EthereumScanner
from bsc import BscScanner
import concurrent.futures
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("crypto_finder_api")

app = Flask(__name__)

# Initialize blockchain scanners
scanners = {
    'ethereum': EthereumScanner,
    'bsc': BscScanner
    # Add other blockchain scanners as they're implemented
    # 'solana': SolanaScanner,
    # etc.
}

@app.route('/api/scan', methods=['GET'])
def scan_blockchains():
    # Get parameters from request
    days = request.args.get('days', default=30, type=int)
    min_profit = request.args.get('min_profit', default=50, type=float)
    chains = request.args.get('chains', default='ethereum', type=str).lower().split(',')
    dev_mode = request.args.get('dev_mode', default='false', type=str).lower() == 'true'
    
    # Validate parameters
    if days <= 0 or min_profit < 0:
        return jsonify({'error': 'Invalid parameters. Days must be positive and min_profit must be non-negative.'}), 400
    
    # Filter to only supported chains
    supported_chains = [chain for chain in chains if chain in scanners]
    if not supported_chains:
        return jsonify({'error': 'No supported blockchains specified', 
                       'supported': list(scanners.keys())}), 400
    
    # Scan blockchains in parallel
    results = []
    errors = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_chain = {}
        
        # Submit scanning tasks
        for chain in supported_chains:
            try:
                scanner = scanners[chain]()
                future = executor.submit(scanner.scan_profitable_wallets, days, min_profit, dev_mode=dev_mode)
                future_to_chain[future] = chain
            except Exception as e:
                errors.append({'chain': chain, 'error': str(e)})
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_chain):
            chain = future_to_chain[future]
            try:
                chain_results = future.result()
                results.extend(chain_results)
            except Exception as e:
                errors.append({'chain': chain, 'error': str(e)})
    
    # Sort by profit percentage (descending)
    results.sort(key=lambda x: x.get('profit_percent', 0), reverse=True)
    
    # Prepare response
    response = {
        'wallets': results,
        'count': len(results),
        'parameters': {
            'days': days,
            'min_profit': min_profit,
            'chains': supported_chains
        }
    }
    
    if errors:
        response['errors'] = errors
        
    return jsonify(response)

@app.route('/api/estimate-time', methods=['GET'])
def estimate_scan_time():
    # Get parameters from request
    days = request.args.get('days', default=30, type=int)
    chains = request.args.get('chains', default='ethereum', type=str).lower().split(',')
    num_samples = request.args.get('samples', default=100, type=int)
    
    # Validate parameters
    if days <= 0:
        return jsonify({'error': 'Invalid parameters. Days must be positive.'}), 400
    
    # Filter to only supported chains
    supported_chains = [chain for chain in chains if chain in scanners]
    if not supported_chains:
        return jsonify({'error': 'No supported blockchains specified', 
                       'supported': list(scanners.keys())}), 400
    
    # Get time estimates for each chain
    estimates = {}
    for chain in supported_chains:
        try:
            scanner = scanners[chain]()
            estimates[chain] = scanner.estimate_scan_time(days, num_samples)
        except Exception as e:
            estimates[chain] = {'error': str(e)}
    
    # Calculate overall estimate (assuming parallel execution)
    if any('min_seconds' in est for est in estimates.values()):
        max_min_time = max([est.get('min_seconds', 0) for est in estimates.values()])
        max_max_time = max([est.get('max_seconds', 0) for est in estimates.values()])
        
        def format_time(seconds):
            if seconds < 60:
                return f"{int(seconds)} seconds"
            elif seconds < 3600:
                return f"{int(seconds/60)} minutes and {int(seconds%60)} seconds"
            else:
                return f"{int(seconds/3600)} hours, {int((seconds%3600)/60)} minutes"
        
        total_estimate = {
            'min_time': format_time(max_min_time),
            'max_time': format_time(max_max_time),
            'parallel_execution': True
        }
    else:
        total_estimate = {
            'error': 'Could not calculate overall time estimate'
        }
    
    return jsonify({
        'overall_estimate': total_estimate,
        'chain_estimates': estimates,
        'parameters': {
            'days': days,
            'chains': supported_chains,
            'num_samples': num_samples
        }
    })

@app.route('/api/supported-chains', methods=['GET'])
def get_supported_chains():
    return jsonify({
        'supported_chains': list(scanners.keys())
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
