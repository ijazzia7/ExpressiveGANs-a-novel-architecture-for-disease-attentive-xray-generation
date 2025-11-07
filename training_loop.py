 def expressive_weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    noise_dim = 100
    lr = 0.0007

    bce_logits = nn.BCEWithLogitsLoss(reduction='mean') 
    bce_logits = nn.BCELoss() 

    λ_disc_cams = 1 
    λ_gan_cams = 5

    teacher = CXR_CAM_Teacher().to(device)
    generator = GeneratorWithCAMGuidance().to(device)
    discriminator = DiscriminatorWithExplain().to(device)

    generator.apply(expressive_weights_init)
    discriminator.apply(expressive_weights_init)

    g_opt = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

Loading Saved

    import torch
    import torch.nn as nn
    import os

    # -------- Initialization Function --------
    def expressive_weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    # -------- Setup --------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    noise_dim = 100
    lr = 0.00002

    bce_logits = nn.BCEWithLogitsLoss(reduction='mean')
    λ_disc_cams = 1
    λ_gan_cams = 5

    # -------- Initialize Models --------
    teacher = CXR_CAM_Teacher().to(device)
    generator = GeneratorWithCAMGuidance().to(device)
    discriminator = DiscriminatorWithExplain().to(device)

    # -------- Initialize Optimizers --------
    g_opt = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # -------- Load Checkpoint if Available --------
    checkpoint_path = "/kaggle/input/nocheckerboardlatest/gan_ckpt_step_no_checkerboard.pt"  # change if needed

    if os.path.exists(checkpoint_path):
        print(f"✅ Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)

        generator.load_state_dict(ckpt['G_state'])
        discriminator.load_state_dict(ckpt['D_state'])
        g_opt.load_state_dict(ckpt['g_opt'])
        d_opt.load_state_dict(ckpt['d_opt'])
        step = ckpt.get('step', 0)

        print(f"Resumed from step {step}")
    else:
        print("⚙️ No checkpoint found. Initializing new models...")
        generator.apply(expressive_weights_init)
        discriminator.apply(expressive_weights_init)
        step = 0

    num_epochs = 50
    #step = 0
    SAVE_EVERY = 90
    LOG_INTERVAL = 50

    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"(Expressive) Epoch {epoch+1}/{num_epochs}")
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            B = imgs.size(0)
            step += 1

            # --------------------- Train Discriminator ---------------------
            set_requires_grad(discriminator, True)
            discriminator.train()
            d_opt.zero_grad()

            z = torch.randn(B, noise_dim, device=device)
            fake_images, attn_maps = generator(z)
            all_images = torch.cat([imgs, fake_images], dim=0)
            logits, disc_cams = discriminator(all_images)

            # with torch.no_grad():
            #     teacher_cams, _, _ = teacher(all_images)

            labels = torch.cat([
                torch.ones_like(logits[:B]),
                torch.zeros_like(logits[:B])
            ], dim=0)

            disc_logits_loss = bce_logits(logits, labels)
            #disc_cams_loss = F.l1_loss(disc_cams, teacher_cams)
            d_loss = disc_logits_loss #+ λ_disc_cams * disc_cams_loss

            d_loss.backward()
            d_opt.step()

            # --------------------- Train Generator ---------------------
            set_requires_grad(discriminator, False)
            generator.train()
            g_opt.zero_grad()

            z = torch.randn(B, noise_dim, device=device)
            fake_images, attn_maps = generator(z)

            # Disc cams for fake images
            logits_fake, disc_cams_fake = discriminator(fake_images)

            # Teacher cams for fake images
            # with torch.no_grad():
            #     teacher_cams_fake, _, _ = teacher(fake_images)

            # Generator wants to: fool D + produce teacher-like CAMs
            g_adv_loss = bce_logits(logits_fake, torch.ones_like(logits_fake))  # fake → 1
            #g_cam_loss = F.l1_loss(disc_cams_fake, teacher_cams_fake)

            # Total G loss
            g_loss = g_adv_loss #+ λ_gan_cams * g_cam_loss
            g_loss.backward()
            g_opt.step()

            set_requires_grad(discriminator, True)

            # --------------------- Logging ---------------------
            if step % 10 == 0:
                pbar.set_postfix({
                    'd_loss': f'{d_loss.item():.4f}',
                    'g_loss': f'{g_loss.item():.4f}',
                })

            if step % LOG_INTERVAL == 0:
                print(f"\n[Expressive] Step {step} Epoch {epoch+1}/{num_epochs}")
                print(f"  D loss: {d_loss.item():.4f}")
                print(f"  G loss: {g_loss.item():.4f}")
                #print(f"  G_adv: {g_adv_loss.item():.4f}, G_cam: {g_cam_loss.item():.4f}")

                print('Generator')
                show_image_tensor(fake_images.detach())
                # show_image_tensor(teacher_cams_fake[:B].detach())
                # show_cams_overlay_batch(fake_images.detach(), teacher_cams_fake[:B].detach(), max_per_batch=5)

            # --------------------- Save Checkpoint ---------------------
            if step % SAVE_EVERY == 0:
                torch.save({
                    'G_state': generator.state_dict(),
                    'D_state': discriminator.state_dict(),
                    'g_opt': g_opt.state_dict(),
                    'd_opt': d_opt.state_dict(),
                    'step': step,
                }, f"gan_ckpt_step.pt")

    z = torch.randn(3, 100, device=device)
    fake_images, attn_maps = generator(z)
    show_image_tensor(fake_images.detach())

    with torch.no_grad():
        teacher_cams_fake, _, _ = teacher(fake_images)
    show_image_tensor(teacher_cams_fake[:B].detach())
