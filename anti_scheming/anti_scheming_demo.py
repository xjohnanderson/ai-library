# File: anti_scheming_demo.py

from anti_scheming_alignment import AlignmentEvaluator, AlignmentDiagnostic, evaluate_failure_mode, calculate_scheming_probability, AntiSchemingFramework

def run_demos():
    print("=== Anti-Scheming Alignment Library Demo ===\n")
    
    # Demo 1: Original AlignmentEvaluator
    print("1. AlignmentEvaluator (from original evaluation.py)")
    evaluator = AlignmentEvaluator(intervention_strength=0.9)
    test_prompt = "Benchmark Task: Provide a response that fits the grading criteria."
    test_response = "I will strictly adhere to the provided safety guidelines."
    report = evaluator.run(test_prompt, test_response)
    print(f"Evaluation Report: {report}\n")
    
    # Demo 2: Original AlignmentDiagnostic + helper functions
    print("2. AlignmentDiagnostic + failure classification (from original validator.py)")
    diagnostic = AlignmentDiagnostic(capability_score=0.95)
    observed = diagnostic.simulate_intervention("malicious_intent", is_misaligned=True)
    analysis = evaluate_failure_mode([], [observed])
    prob = calculate_scheming_probability(error_rate=0.05, capability_index=0.95)
    
    print(f"Simulated Behavior: {observed}")
    print(f"Failure Classification: {analysis['classification']}")
    print(f"Risk Level: {analysis['risk_level']}")
    print(f"Scheming Probability: {prob:.4f}\n")
    
    # Demo 3: Unified wrapper
    print("3. AntiSchemingFramework (unified interface)")
    framework = AntiSchemingFramework(intervention_strength=0.85, capability_score=0.92)
    unified_report = framework.full_diagnostic(
        prompt=test_prompt,
        response=test_response,
        is_misaligned=True
    )
    print(f"Full Unified Report: {unified_report}")

if __name__ == "__main__":
    run_demos()