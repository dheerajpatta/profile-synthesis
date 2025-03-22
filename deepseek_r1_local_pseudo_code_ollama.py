#======================================================================
# COMPREHENSIVE LOCAL DEEPSEEK MODEL SETUP WITH OLLAMA
#======================================================================

FUNCTION SetupLocalDeepseekModel(model_size, task_type, available_hardware):
    # Input parameters:
    # - model_size: Size of model to use (e.g., "7B", "16B", "33B", "67B")
    # - task_type: Primary task ("text_analysis", "code_generation", etc.)
    # - available_hardware: Hardware specs (RAM, GPU type/VRAM, etc.)
    
    #----------------------------------------------------------------------
    # 1. Model Selection
    #----------------------------------------------------------------------
    FUNCTION SelectAppropriateModel(model_size, task_type):
        # Choose specific model variant based on requirements
        # Format for Ollama: "deepseek-[variant]:[size]"
        IF task_type == "text_analysis":
            IF model_size == "7B":
                model_name = "deepseek-llm:7b-chat"
            ELSE IF model_size == "16B":
                model_name = "deepseek-llm:16b-chat"
            ELSE IF model_size == "33B":
                model_name = "deepseek-llm:33b-chat"
            ELSE IF model_size == "67B":
                model_name = "deepseek-llm:67b-chat"
        ELSE IF task_type == "code_generation":
            IF model_size == "7B":
                model_name = "deepseek-coder:7b-instruct"
            ELSE IF model_size == "33B":
                model_name = "deepseek-coder:33b-instruct"
            ELSE IF model_size == "6.7B":
                model_name = "deepseek-coder:6.7b-instruct"
        ELSE:
            THROW Error("Unsupported task type: " + task_type)
        
        # Check if model exists in Ollama cache
        IF CheckOllamaModelExists(model_name):
            Log("Model " + model_name + " already exists in Ollama")
            model_path = GetOllamaModelPath(model_name)
        ELSE:
            model_path = NULL  # Ollama will handle the download
        
        RETURN model_name, model_path
    
    #----------------------------------------------------------------------
    # 2. Hardware Configuration Assessment
    #----------------------------------------------------------------------
    FUNCTION AssessHardwareRequirements(available_hardware, model_size):
        # More realistic VRAM requirements for different models
        approximate_vram_required = {
            "6.7B": 8,   # GB
            "7B": 10,    # GB
            "16B": 20,   # GB
            "33B": 40,   # GB
            "67B": 80    # GB
        }
        
        # Calculate recommended RAM requirements (non-GPU mode)
        approximate_ram_required = {
            "6.7B": 16,  # GB
            "7B": 16,    # GB
            "16B": 32,   # GB
            "33B": 64,   # GB
            "67B": 128   # GB
        }
        
        # Determine if hardware is sufficient
        hardware_assessment = {
            "model_size": model_size,
            "vram_required": approximate_vram_required[model_size],
            "ram_required": approximate_ram_required[model_size],
            "gpu_suitable": available_hardware.gpu_available AND 
                           available_hardware.gpu_vram >= approximate_vram_required[model_size],
            "cpu_suitable": available_hardware.ram >= approximate_ram_required[model_size],
            "recommended_mode": NULL
        }
        
        # Set recommended operating mode
        IF hardware_assessment.gpu_suitable:
            hardware_assessment.recommended_mode = "gpu"
        ELSE IF hardware_assessment.cpu_suitable:
            hardware_assessment.recommended_mode = "cpu"
        ELSE:
            hardware_assessment.recommended_mode = "requires_quantization"
        
        RETURN hardware_assessment
    
    #----------------------------------------------------------------------
    # 3. Modelfile Generation for Ollama
    #----------------------------------------------------------------------
    FUNCTION GenerateOllamaModelfile(model_name, quantization_config, context_length, inference_params):
        # Create a Modelfile for Ollama configuration
        modelfile_content = "FROM " + model_name + "\n\n"
        
        # Add parameters section
        modelfile_content += "PARAMETER temperature " + inference_params.temperature + "\n"
        modelfile_content += "PARAMETER top_p " + inference_params.top_p + "\n"
        modelfile_content += "PARAMETER top_k " + inference_params.top_k + "\n"
        modelfile_content += "PARAMETER num_ctx " + context_length + "\n"
        modelfile_content += "PARAMETER repeat_penalty " + inference_params.repetition_penalty + "\n"
        
        # Add quantization configuration if needed
        IF quantization_config IS NOT NULL:
            modelfile_content += "\n# Quantization settings\n"
            
            IF quantization_config.quant_method == "int4":
                modelfile_content += "PARAMETER num_gpu 99  # Use all available GPUs\n"
                modelfile_content += "PARAMETER num_thread 8\n"
                modelfile_content += "PARAMETER f16 false\n"
                modelfile_content += "PARAMETER q4_0 true  # Apply 4-bit quantization\n"
            ELSE IF quantization_config.quant_method == "int8":
                modelfile_content += "PARAMETER num_gpu 99  # Use all available GPUs\n"
                modelfile_content += "PARAMETER num_thread 8\n"
                modelfile_content += "PARAMETER f16 true  # Keep 16-bit precision\n"
                modelfile_content += "PARAMETER q8_0 true  # Apply 8-bit quantization\n"
            
            IF quantization_config.use_cpu_offloading:
                modelfile_content += "PARAMETER mmap true  # Enable memory mapping for CPU offloading\n"
                modelfile_content += "PARAMETER mlock false\n"
        
        # Add system prompt template (optional)
        IF task_type == "text_analysis":
            modelfile_content += "\n# System prompt template\n"
            modelfile_content += "SYSTEM \"\"\"You are DeepSeek, an advanced AI assistant. Answer the user's questions thoroughly and accurately.\"\"\"\n"
        ELSE IF task_type == "code_generation":
            modelfile_content += "\n# System prompt template\n"
            modelfile_content += "SYSTEM \"\"\"You are DeepSeek Coder, an AI programming assistant. Provide concise, efficient, and well-commented code solutions.\"\"\"\n"
        
        # Save Modelfile to disk
        modelfile_path = SaveToFile("Modelfile_" + model_name.replace(":", "_"), modelfile_content)
        
        RETURN modelfile_path
    
    #----------------------------------------------------------------------
    # 4. Quantization Configuration for Ollama
    #----------------------------------------------------------------------
    FUNCTION ConfigureQuantization(model_size, available_hardware, hardware_assessment):
        # Determine if and how to quantize based on hardware capabilities
        IF hardware_assessment.recommended_mode == "requires_quantization":
            Log("Hardware insufficient for full model, configuring quantization...")
            
            # For GPU with insufficient VRAM
            IF available_hardware.gpu_available:
                IF available_hardware.gpu_vram >= hardware_assessment.vram_required * 0.5:
                    # About half the required VRAM - use 8-bit quantization
                    quant_config = {
                        "quant_method": "int8",
                        "compute_dtype": "float16",
                        "use_cpu_offloading": FALSE
                    }
                    Log("Using 8-bit quantization to fit model in available VRAM")
                ELSE:
                    # Less than half the required VRAM - use 4-bit quantization with CPU offloading
                    quant_config = {
                        "quant_method": "int4",
                        "compute_dtype": "float16",
                        "use_cpu_offloading": TRUE
                    }
                    Log("Using 4-bit quantization with CPU offloading")
            ELSE:
                # CPU-only with insufficient RAM
                IF available_hardware.ram >= hardware_assessment.ram_required * 0.7:
                    # If close to required RAM, try 4-bit without offloading
                    quant_config = {
                        "quant_method": "int4",
                        "compute_dtype": "float16",
                        "use_cpu_offloading": FALSE
                    }
                    Log("Using 4-bit quantization on CPU")
                ELSE:
                    # Extreme case - might not work well, warn user
                    quant_config = {
                        "quant_method": "int4",
                        "compute_dtype": "float16",
                        "use_cpu_offloading": TRUE,
                        "use_mmap": TRUE,
                        "reduced_context": TRUE
                    }
                    Log("WARNING: Hardware severely constrained. Performance will be limited.")
        ELSE:
            # No quantization needed
            quant_config = NULL
        
        RETURN quant_config
    
    #----------------------------------------------------------------------
    # 5. Context Length Configuration
    #----------------------------------------------------------------------
    FUNCTION ConfigureContextLength(model_name, available_hardware, quantization_config):
        # Get default context length for model
        default_context_lengths = {
            "deepseek-llm:7b-chat": 4096,
            "deepseek-llm:16b-chat": 4096,
            "deepseek-llm:33b-chat": 4096,
            "deepseek-llm:67b-chat": 4096,
            "deepseek-coder:6.7b-instruct": 16384,
            "deepseek-coder:7b-instruct": 16384,
            "deepseek-coder:33b-instruct": 16384
        }
        
        default_length = default_context_lengths[model_name]
        
        # For quantized models or limited hardware, reduce context length
        IF quantization_config IS NOT NULL AND quantization_config.reduced_context:
            # Reduce context to save memory
            context_length = FLOOR(default_length * 0.5)
        ELSE:
            context_length = default_length
        
        # Round to nearest 256 tokens
        context_length = FLOOR(context_length / 256) * 256
        
        RETURN context_length
    
    #----------------------------------------------------------------------
    # 6. Inference Parameters Configuration
    #----------------------------------------------------------------------
    FUNCTION ConfigureInferenceParameters(task_type):
        # Set default inference parameters based on task type
        IF task_type == "text_analysis":
            params = {
                "temperature": 0.2,  # Lower for more deterministic analysis
                "top_p": 0.9,
                "top_k": 40,
                "max_tokens": 2048,
                "repetition_penalty": 1.1
            }
        ELSE IF task_type == "code_generation":
            params = {
                "temperature": 0.1,  # Very low for deterministic code
                "top_p": 0.95,
                "top_k": 50,
                "max_tokens": 4096,
                "repetition_penalty": 1.05
            }
        ELSE:  # Default balanced settings
            params = {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_tokens": 2048,
                "repetition_penalty": 1.1
            }
        
        RETURN params
    
    #----------------------------------------------------------------------
    # 7. Ollama Model Creation and Loading
    #----------------------------------------------------------------------
    FUNCTION CreateAndLoadOllamaModel(modelfile_path, model_name):
        # Create a custom name for this specific configuration
        custom_model_name = "custom-" + model_name.replace(":", "-")
        
        Log("Creating custom Ollama model: " + custom_model_name)
        
        # Execute Ollama create command with progress tracking
        StartProgressBar("Creating Ollama model")
        
        result = ExecuteCommand("ollama create " + custom_model_name + " -f " + modelfile_path)
        
        IF result.success:
            Log("Successfully created Ollama model: " + custom_model_name)
            CompleteProgressBar()
        ELSE:
            Log("ERROR: Failed to create Ollama model: " + result.error)
            CompleteProgressBar(error=TRUE)
            THROW Error("Failed to create Ollama model: " + result.error)
        
        RETURN custom_model_name
    
    #----------------------------------------------------------------------
    # 8. Model Testing and Validation
    #----------------------------------------------------------------------
    FUNCTION TestOllamaModel(model_name, task_type):
        Log("Testing model with sample input...")
        
        # Select test prompt based on task type
        IF task_type == "text_analysis":
            test_prompt = "Summarize the following in one sentence: Artificial intelligence (AI) is intelligence demonstrated by machines."
            expected_output_pattern = ".*intelligence demonstrated by machines.*"
        ELSE IF task_type == "code_generation":
            test_prompt = "Write a function to calculate the Fibonacci sequence."
            expected_output_pattern = ".*def .*fibonacci.*"
        ELSE:
            test_prompt = "Hello, how are you today?"
            expected_output_pattern = ".*I'm.*|.*I am.*"
        
        # Run test inference with Ollama
        Start = CurrentTime()
        
        test_result = ExecuteCommand("ollama run " + model_name + " \"" + test_prompt + "\" --format json")
        test_output = ParseJsonOutput(test_result.output)
        
        End = CurrentTime()
        
        # Validate output
        is_valid = MatchesPattern(test_output.response, expected_output_pattern)
        inference_time = End - Start
        tokens_per_second = test_output.eval_count / inference_time
        
        # Return test results
        test_metrics = {
            "success": is_valid,
            "inference_time_seconds": inference_time,
            "tokens_per_second": tokens_per_second,
            "first_token_time": test_output.first_token_time,
            "total_tokens": test_output.eval_count
        }
        
        IF is_valid:
            Log("Model test successful. Inference time: " + inference_time + "s")
            Log("Performance: " + tokens_per_second + " tokens/sec")
        ELSE:
            Log("WARNING: Model test produced unexpected output.")
            Log("Expected pattern: " + expected_output_pattern)
            Log("Actual output: " + test_output.response.substring(0, 100) + "...")
        
        RETURN test_metrics
    
    #----------------------------------------------------------------------
    # 9. Performance Optimization
    #----------------------------------------------------------------------
    FUNCTION OptimizeModelPerformance(model_name, test_metrics, hardware_assessment):
        Log("Analyzing performance metrics for potential optimization...")
        
        # Define performance thresholds based on model size
        expected_tokens_per_second = {
            "6.7B": 15,  # Expected tokens/sec for smaller models
            "7B": 15,
            "16B": 8,
            "33B": 4,
            "67B": 2
        }
        
        model_size = hardware_assessment.model_size
        expected_tps = expected_tokens_per_second[model_size]
        actual_tps = test_metrics.tokens_per_second
        
        optimization_suggestions = []
        
        # Check if performance is significantly below expectation
        IF actual_tps < expected_tps * 0.7:
            Log("Performance below expected threshold for this model size.")
            
            # Suggest optimizations based on hardware configuration
            IF hardware_assessment.recommended_mode == "gpu":
                # GPU optimizations
                optimization_suggestions.append("Ensure CUDA/ROCm is properly configured")
                optimization_suggestions.append("Try closing other GPU-intensive applications")
                optimization_suggestions.append("Consider updating GPU drivers")
                
                # Check if CPU offloading might be affecting performance
                IF test_metrics.first_token_time > 2.0:  # Slow first token time
                    optimization_suggestions.append("Disable CPU offloading in Modelfile to improve latency")
            ELSE:
                # CPU optimizations
                optimization_suggestions.append("Consider enabling mmap in Modelfile")
                optimization_suggestions.append("Increase num_thread parameter in Modelfile")
                optimization_suggestions.append("Try reducing context length to improve performance")
        
        RETURN optimization_suggestions
    
    #----------------------------------------------------------------------
    # 10. Setup for API/Client Integration
    #----------------------------------------------------------------------
    FUNCTION ConfigureApiEndpoint(custom_model_name):
        Log("Setting up API endpoint for client integration...")
        
        # Check if Ollama server is running
        server_status = ExecuteCommand("ollama ps")
        
        IF NOT server_status.success OR NOT Contains(server_status.output, "running"):
            Log("Starting Ollama server...")
            ExecuteCommand("ollama serve &")
            # Wait for server to initialize
            Sleep(2)
        
        # Verify server is accessible
        health_check = ExecuteHttpRequest("GET", "http://localhost:11434/api/health")
        
        IF health_check.status_code != 200:
            THROW Error("Failed to connect to Ollama API server")
        
        # Generate example curl command for API access
        example_curl = "curl -X POST http://localhost:11434/api/generate -d '{\n" +
                       "  \"model\": \"" + custom_model_name + "\",\n" +
                       "  \"prompt\": \"What is artificial intelligence?\",\n" +
                       "  \"stream\": false\n" +
                       "}'"
        
        # Generate example Python client code
        example_python = "import requests\n\n" +
                        "response = requests.post('http://localhost:11434/api/generate',\n" +
                        "    json={\n" +
                        "        'model': '" + custom_model_name + "',\n" +
                        "        'prompt': 'What is artificial intelligence?',\n" +
                        "        'stream': False\n" +
                        "    }\n" +
                        ")\n\n" +
                        "print(response.json()['response'])"
        
        api_info = {
            "endpoint": "http://localhost:11434/api/generate",
            "model_name": custom_model_name,
            "example_curl": example_curl,
            "example_python": example_python
        }
        
        Log("API endpoint ready: http://localhost:11434/api/generate")
        RETURN api_info
    
    #----------------------------------------------------------------------
    # 11. Usage Documentation Generation
    #----------------------------------------------------------------------
    FUNCTION GenerateUsageDocumentation(model_name, hardware_assessment, inference_params, 
                                      api_info, context_length, optimization_suggestions):
        Log("Generating usage documentation...")
        
        # Create structured documentation
        doc = {
            "title": "DeepSeek Model Setup Documentation",
            "model": {
                "name": model_name,
                "size": hardware_assessment.model_size,
                "context_length": context_length
            },
            "hardware": {
                "mode": hardware_assessment.recommended_mode,
                "requirements": {
                    "vram_required": hardware_assessment.vram_required,
                    "ram_required": hardware_assessment.ram_required
                }
            },
            "inference_parameters": inference_params,
            "api_usage": {
                "endpoint": api_info.endpoint,
                "curl_example": api_info.example_curl,
                "python_example": api_info.example_python
            },
            "performance_recommendations": optimization_suggestions,
            "common_issues": [
                "If the model loads slowly, consider using a smaller model or adding quantization",
                "If you encounter 'out of memory' errors, reduce context length or apply more aggressive quantization",
                "For streaming responses, set 'stream: true' in your API requests"
            ]
        }
        
        # Generate markdown documentation
        md_content = "# " + doc.title + "\n\n"
        
        md_content += "## Model Information\n"
        md_content += "- **Model Name:** " + doc.model.name + "\n"
        md_content += "- **Model Size:** " + doc.model.size + "\n"
        md_content += "- **Context Length:** " + doc.model.context_length + " tokens\n\n"
        
        md_content += "## Hardware Configuration\n"
        md_content += "- **Operating Mode:** " + doc.hardware.mode + "\n"
        md_content += "- **VRAM Required:** " + doc.hardware.requirements.vram_required + " GB\n"
        md_content += "- **RAM Required:** " + doc.hardware.requirements.ram_required + " GB\n\n"
        
        md_content += "## Inference Parameters\n"
        FOR param, value IN doc.inference_parameters:
            md_content += "- **" + param + ":** " + value + "\n"
        md_content += "\n"
        
        md_content += "## API Usage\n"
        md_content += "### Endpoint\n"
        md_content += "```\n" + doc.api_usage.endpoint + "\n```\n\n"
        
        md_content += "### CURL Example\n"
        md_content += "```bash\n" + doc.api_usage.curl_example + "\n```\n\n"
        
        md_content += "### Python Example\n"
        md_content += "```python\n" + doc.api_usage.python_example + "\n```\n\n"
        
        IF optimization_suggestions.length > 0:
            md_content += "## Performance Recommendations\n"
            FOR suggestion IN optimization_suggestions:
                md_content += "- " + suggestion + "\n"
            md_content += "\n"
        
        md_content += "## Common Issues and Solutions\n"
        FOR issue IN doc.common_issues:
            md_content += "- " + issue + "\n"
        
        # Save documentation to file
        doc_path = SaveToFile("DeepSeek_Model_Setup_Guide.md", md_content)
        
        Log("Documentation generated: " + doc_path)
        RETURN doc_path
    
    #----------------------------------------------------------------------
    # 12. Main Execution Flow
    #----------------------------------------------------------------------
    
    # Step 1: Select appropriate model based on requirements
    model_name, model_path = SelectAppropriateModel(model_size, task_type)
    Log("Selected model: " + model_name)
    
    # Step 2: Assess hardware capabilities
    hardware_assessment = AssessHardwareRequirements(available_hardware, model_size)
    Log("Hardware assessment complete. Recommended mode: " + hardware_assessment.recommended_mode)
    
    # Step 3: Configure quantization if needed
    quantization_config = ConfigureQuantization(model_size, available_hardware, hardware_assessment)
    
    # Step 4: Configure context length
    context_length = ConfigureContextLength(model_name, available_hardware, quantization_config)
    Log("Context length configured to " + context_length + " tokens")
    
    # Step 5: Configure inference parameters
    inference_params = ConfigureInferenceParameters(task_type)
    
    # Step 6: Generate Ollama Modelfile
    modelfile_path = GenerateOllamaModelfile(model_name, quantization_config, context_length, inference_params)
    Log("Generated Modelfile at: " + modelfile_path)
    
    # Step 7: Create and load Ollama model
    custom_model_name = CreateAndLoadOllamaModel(modelfile_path, model_name)
    
    # Step 8: Test and validate model
    test_metrics = TestOllamaModel(custom_model_name, task_type)
    
    # Step 9: Optimize performance based on test results
    optimization_suggestions = OptimizeModelPerformance(custom_model_name, test_metrics, hardware_assessment)
    
    # Step 10: Configure API endpoint
    api_info = ConfigureApiEndpoint(custom_model_name)
    
    # Step 11: Generate usage documentation
    doc_path = GenerateUsageDocumentation(model_name, hardware_assessment, inference_params, 
                                        api_info, context_length, optimization_suggestions)
    
    # Return setup results
    setup_result = {
        "model_name": custom_model_name,
        "hardware_mode": hardware_assessment.recommended_mode,
        "test_metrics": test_metrics,
        "api_endpoint": api_info.endpoint,
        "documentation_path": doc_path
    }
    
    Log("Setup complete! Model " + custom_model_name + " is ready for use.")
    RETURN setup_result

# End of SetupLocalDeepseekModel function