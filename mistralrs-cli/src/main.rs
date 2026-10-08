            )
            .await?;
        }

        Command::Run {
            model_type,
            default_model,
            runtime,
            agent_options,
            sandbox,
            thinking,
            reasoning_effort,
            input,
            max_tokens,
            image,
            video,
            audio,
            adapter,
        } => {
            let model_type = resolve_model_type(model_type, default_model)?;
            run_interactive(
                model_type,
                runtime,
                agent_options,
                sandbox,
                cli.global,
                thinking,
                reasoning_effort,
                input,
                max_tokens,
                image,
                video,
                audio,
                adapter,
            )
            .await?;
        }

        Command::Stream {
            base_url,
            model,
            input,
            max_tokens,
            temperature,
            top_p,