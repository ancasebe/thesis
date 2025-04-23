import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gui.research_members.climber_db_manager import ClimberDatabaseManager
from gui.test_page.test_db_manager import ClimbingTestManager

def plot_normalized_max_force(climber_manager, test_manager, admin_id, current_test_id=None):
    """
    Plot normalized max strength data against climbing grade (IRCRA).
    
    1) Scatter plot with regression line
    2) Histogram of normalized max strength values
    
    Only uses tests with valid max_strength, weight, and IRCRA values.
    """
    # Get all tests from database
    all_tests = test_manager.fetch_results_by_admin(str(admin_id))
    print(f"Retrieved {len(all_tests)} tests from database")
    
    # Process tests to find valid ones with all required data
    valid_tests = []
    
    for test in all_tests:
        try:
            test_id = test['id']
            participant_id = test.get('participant_id')
            
            # Skip if missing participant ID
            if not participant_id:
                print(f"Test ID {test_id}: Missing participant ID")
                continue
                
            # Try to get test_results - handle as string or dict
            test_results = None
            if isinstance(test.get('test_results'), dict):
                test_results = test.get('test_results')
            elif isinstance(test.get('test_results'), str):
                try:
                    test_results = json.loads(test.get('test_results') or "{}")
                except json.JSONDecodeError:
                    print(f"Test ID {test_id}: Invalid JSON in test_results")
                    continue
            
            # Skip if no test_results or missing max_strength
            if not test_results or 'max_strength' not in test_results:
                print(f"Test ID {test_id}: Missing max_strength data")
                continue
                
            # Get user data
            user = climber_manager.get_user_data(str(admin_id), participant_id)
            
            # Skip if user is missing weight
            if not user or not user.get('weight'):
                print(f"Test ID {test_id}: User missing weight data")
                continue
                
            # Skip if IRCRA is missing or not a number
            ircra = user.get('ircra')
            if not ircra or ircra == 'N/A':
                print(f"Test ID {test_id}: User missing IRCRA grade")
                continue
                
            # Try to convert IRCRA to a number
            try:
                ircra = float(ircra)
            except (ValueError, TypeError):
                print(f"Test ID {test_id}: Invalid IRCRA grade format")
                continue
                
            # Calculate normalized force
            weight = float(user['weight'])
            max_strength = float(test_results['max_strength'])
            norm_max_force = max_strength / weight
            
            # Add to valid tests
            valid_tests.append({
                'test_id': test_id,
                'ircra': ircra,
                'norm_max_force': norm_max_force,
                'max_strength': max_strength,
                'weight': weight
            })
            print(f"Test ID {test_id}: Valid data (IRCRA={ircra}, Weight={weight}, Max strength={max_strength})")
            
        except Exception as e:
            print(f"Error processing test {test.get('id')}: {e}")
    
    # Create DataFrame from valid tests
    df = pd.DataFrame(valid_tests)
    
    # Check if we have any valid data
    if df.empty:
        print("No valid tests found with all required data (max_strength, weight, IRCRA).")
        return
    
    print(f"\nFound {len(df)} valid tests with all required data")
    print(f"Valid test IDs: {df['test_id'].tolist()}")
    
    # If no current_test_id specified or the specified one isn't valid,
    # use the first valid test instead
    if current_test_id is None or current_test_id not in df['test_id'].values:
        current_test_id = df['test_id'].iloc[0]
        print(f"Using test ID {current_test_id} as current test")
    
    # Get the current test data
    curr = df[df['test_id'] == current_test_id].iloc[0]
    
    # PLOT 1: Scatter plot with regression line
    if len(df) >= 2:  # Need at least 2 points for regression
        x = df['ircra']
        y = df['norm_max_force']
        
        # Calculate regression line
        slope, intercept = np.polyfit(x, y, 1)
        line = np.poly1d((slope, intercept))
        
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, alpha=0.7, edgecolor='k', s=80, label='All tests')
        
        # Plot regression line
        xs = np.linspace(x.min(), x.max(), 100)
        plt.plot(xs, line(xs), linestyle='--', linewidth=2, color='orange',
                 label=f'Trend line (slope={slope:.3f})')
        
        # Highlight current test
        plt.scatter(curr['ircra'], curr['norm_max_force'], marker='D', s=150, 
                    color='red', label=f'Current test (ID={current_test_id})')
        
        plt.xlabel('Climbing Grade (IRCRA)', fontsize=12)
        plt.ylabel('Max Force / Body Weight', fontsize=12)
        plt.title('Normalized Max Force vs. Climbing Grade', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough data points for regression analysis (need at least 2)")
    
    # PLOT 2: Histogram with normal curve and current test marker
    if len(df) >= 3:  # Need at least 3 points for histogram
        y = df['norm_max_force']
        
        plt.figure(figsize=(10, 6))
        counts, bins, _ = plt.hist(y, bins=min(10, len(df)), 
                                    color='skyblue', alpha=0.7, edgecolor='black')
        
        # Only add normal curve if we have enough data points
        if len(df) >= 5:
            bin_width = bins[1] - bins[0]
            mu, sigma = y.mean(), y.std()
            x = np.linspace(min(bins), max(bins), 100)
            pdf = (1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2))
            plt.plot(x, pdf * len(y) * bin_width, 'r-', linewidth=2, 
                     label=f'Normal distribution\n(mean={mu:.2f}, std={sigma:.2f})')
        
        # Add vertical line for current test
        plt.axvline(curr['norm_max_force'], color='red', linestyle='-', linewidth=2,
                    label=f'Current test: {curr["norm_max_force"]:.2f}')
        
        plt.xlabel('Max Force / Body Weight', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Distribution of Normalized Max Force', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough data points for histogram (need at least 3)")

if __name__ == '__main__':
    try:
        # Create database managers
        climber_manager = ClimberDatabaseManager()
        test_manager = ClimbingTestManager()
        
        # Use test ID 55 which appears to have all required data
        current_test_id = 55
        
        # Run the plotting function
        plot_normalized_max_force(
            climber_manager,
            test_manager,
            admin_id=1,
            current_test_id=current_test_id
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Make sure to close database connections
        try:
            climber_manager.close()
            test_manager.close_connection()
        except:
            pass