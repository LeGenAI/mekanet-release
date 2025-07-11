#!/usr/bin/env python3
"""
Main execution script for TESSD paper reproduction
Simple wrapper for easy experiment execution
"""

import sys
import argparse
from pathlib import Path

# Add experiments directory to path
sys.path.append(str(Path(__file__).parent / "experiments" / "detection"))

from paper_reproduction_runner import PaperReproductionRunner


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="TESSD Paper Reproduction Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run
  python run_paper_reproduction.py --quick --dry-run
  
  # Full reproduction
  python run_paper_reproduction.py --config configs/paper_reproduction_full.yaml
  
  # Resume interrupted run
  python run_paper_reproduction.py --resume
  
  # Run specific experiments only
  python run_paper_reproduction.py --config configs/paper_reproduction_quick.yaml --dry-run
        """
    )
    
    # Configuration options
    parser.add_argument("--config", 
                       help="Configuration file path")
    parser.add_argument("--quick", action="store_true",
                       help="Use quick test configuration")
    parser.add_argument("--output", default="./paper_reproduction_results",
                       help="Output directory")
    
    # Execution options
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint")
    parser.add_argument("--fail-fast", action="store_true", 
                       help="Stop on first failure")
    parser.add_argument("--dry-run", action="store_true",
                       help="Simulate execution")
    
    # Advanced options
    parser.add_argument("--checkpoint", 
                       help="Custom checkpoint file")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Determine configuration file
    if args.quick:
        config_file = "experiments/detection/configs/paper_reproduction_quick.yaml"
    elif args.config:
        config_file = args.config
    else:
        config_file = "experiments/detection/configs/paper_reproduction_full.yaml"
    
    # Set output directory for quick tests
    if args.quick and args.output == "./paper_reproduction_results":
        args.output = "./quick_test_results"
    
    print("🔬 TESSD Paper Reproduction Runner")
    print(f"📁 Config: {config_file}")
    print(f"📂 Output: {args.output}")
    print(f"🔄 Resume: {args.resume}")
    print(f"🏃 Dry run: {args.dry_run}")
    print()
    
    # Initialize and run
    try:
        runner = PaperReproductionRunner(
            config_file=config_file,
            output_dir=args.output,
            checkpoint_file=args.checkpoint
        )
        
        results = runner.run_all_experiments(
            resume=args.resume,
            fail_fast=args.fail_fast,
            dry_run=args.dry_run
        )
        
        # Print summary
        successful = sum(1 for r in results.values() if r.get('status') == 'success')
        total = len(results)
        
        print("\n" + "="*50)
        print("🎉 PAPER REPRODUCTION COMPLETED!")
        print("="*50)
        print(f"✅ Success rate: {successful}/{total} ({successful/total*100:.1f}%)")
        print(f"📁 Results: {args.output}")
        
        if successful < total:
            print(f"\n❌ Failed experiments ({total - successful}):")
            for name, result in results.items():
                if result.get('status') != 'success':
                    print(f"  - {name}: {result.get('error_message', 'Unknown error')}")
        
        return 0 if successful == total else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 