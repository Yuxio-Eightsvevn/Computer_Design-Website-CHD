/**
 * Video3DModal - 视频弹窗3D可视化模块
 * 用于在diagnosis.html的videoModal中显示3D切片
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

class Video3DModal {
    constructor(canvasElement, options = {}) {
        this.canvas = canvasElement;
        this.videoElement = options.videoElement;
        this.framesDir = options.framesDir;
        this.metadataPath = options.metadataPath; // 新增：metadata.json路径
        this.frameCount = options.frameCount || 31;
        this.fps = options.fps || 15.15;
        this.highlightFrame = options.highlightFrame || 10;

        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.composer = null;
        this.cards = [];
        this.frameTextures = [];
        this.currentHighlightIndex = -1;
        this.boundingBoxes = []; // 新增：存储红色矩形框

        // 新增：从metadata读取的ROI裁剪框数据
        this.roiCropBox = null; // [x1, y1, x2, y2]
        this.imageWidth = 0;
        this.imageHeight = 0;

        this.isInitialized = false;
    }

    async init() {
        try {
            console.log('🎬 初始化3D显示...');

            // 初始化场景
            this.initScene();

            // 读取metadata获取ROI裁剪框信息
            await this.loadMetadata();

            // 加载帧图片
            await this.loadFrames();

            // 创建3D卡片切片（包含红色矩形框）
            this.createCardStack();

            // 绑定视频事件
            this.bindVideoEvents();

            // 启动动画循环
            this.animate();

            this.isInitialized = true;
            console.log('✅ 3D显示初始化完成');

        } catch (error) {
            console.error('❌ 3D显示初始化失败:', error);
            throw error;
        }
    }

    initScene() {
        // 创建场景
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a0a2e);
        this.scene.fog = new THREE.Fog(0x1a0a2e, 10, 50);

        // 创建相机
        const aspect = this.canvas.clientWidth / this.canvas.clientHeight;
        this.camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
        this.camera.position.set(0, 2, 10);
        this.camera.lookAt(0, 0, 0);

        // 创建渲染器
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.2;

        // 添加灯光
        this.initLights();

        // 添加控制器
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.maxPolarAngle = Math.PI / 2 + 0.2;
        this.controls.minPolarAngle = Math.PI / 2 - 0.8;
        this.controls.minDistance = 5;
        this.controls.maxDistance = 5;
        this.controls.target.set(0, 0, 0);

        // 添加后处理效果
        this.composer = new EffectComposer(this.renderer);
        const renderPass = new RenderPass(this.scene, this.camera);
        this.composer.addPass(renderPass);

        const bloomPass = new UnrealBloomPass(
            new THREE.Vector2(this.canvas.clientWidth, this.canvas.clientHeight),
            1.5, 0.4, 0.85
        );
        this.composer.addPass(bloomPass);
    }

    initLights() {
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);

        const mainLight = new THREE.DirectionalLight(0xffffff, 0.8);
        mainLight.position.set(5, 10, 5);
        this.scene.add(mainLight);

        const rimLight = new THREE.PointLight(0x00ffff, 1, 30);
        rimLight.position.set(-5, 5, 10);
        this.scene.add(rimLight);
    }

    /**
     * 从metadata.json加载ROI裁剪框信息
     */
    async loadMetadata() {
        if (!this.metadataPath) {
            console.log('ℹ️ 未提供metadata路径，将不显示红色矩形框');
            return;
        }

        try {
            console.log('📄 读取metadata.json...');
            const response = await fetch(this.metadataPath);

            if (!response.ok) {
                console.warn('⚠️ 无法加载metadata.json，将不显示红色矩形框');
                return;
            }

            const metadata = await response.json();

            // 读取roi_crop_box_original_space参数
            if (metadata.roi_crop_box_original_space && Array.isArray(metadata.roi_crop_box_original_space)) {
                this.roiCropBox = metadata.roi_crop_box_original_space;
                console.log(`✅ 从metadata获取ROI裁剪框: [${this.roiCropBox.join(', ')}]`);
            } else {
                console.warn('⚠️ metadata中没有roi_crop_box_original_space字段');
            }

            // 读取图片尺寸（用于坐标转换）
            if (metadata.width) this.imageWidth = metadata.width;
            if (metadata.height) this.imageHeight = metadata.height;

            console.log(`📐 图片尺寸: ${this.imageWidth}x${this.imageHeight}`);

        } catch (error) {
            console.warn('⚠️ 读取metadata失败:', error, '将不显示红色矩形框');
        }
    }

    /**
     * 创建红色矩形框（真3D立方体线框）
     * @param {number} cardWidth - 卡片宽度
     * @param {number} cardHeight - 卡片高度
     * @param {number} zPosition - Z轴位置
     * @returns {THREE.Group} 线框组
     */
    createRedBoundingBox(cardWidth, cardHeight, zPosition) {
        // 如果没有ROI裁剪框数据，不创建矩形框
        if (!this.roiCropBox || this.roiCropBox.length !== 4) {
            console.warn('⚠️ 没有ROI裁剪框数据，跳过创建红色矩形框');
            return null;
        }

        // roi_crop_box_original_space格式：[x1, y1, x2, y2]
        // 其中(x1,y1)是左上角，(x2,y2)是右下角
        const [x1, y1, x2, y2] = this.roiCropBox;

        // 如果没有图片尺寸，使用默认值（16:9比例）
        const imgWidth = this.imageWidth || 400;
        const imgHeight = this.imageHeight || 225;

        // 定义矩形的4个角点（图片坐标系，左上角为原点）
        const corners = [
            { x: x1, y: y2 },  // 左下 (x1, y2)
            { x: x2, y: y2 },  // 右下 (x2, y2)
            { x: x2, y: y1 },  // 右上 (x2, y1)
            { x: x1, y: y1 }   // 左上 (x1, y1)
        ];

        // 转换到Three.js坐标系（中心为原点，Y轴向上）
        const normalized = corners.map(c => ({
            x: (c.x / imgWidth - 0.5) * cardWidth,
            y: -(c.y / imgHeight - 0.5) * cardHeight  // Y轴反转
        }));

        const group = new THREE.Group();
        const lineThickness = 0.012; // 线条宽度(横截面)
        const lineDepth = 0.1;      // 线条深度(Z轴方向的厚度)

        // 创建红色材质(带光照效果)
        const material = new THREE.MeshStandardMaterial({
            color: 0xff0000,
            transparent: true,
            opacity: 0.9,
            emissive: 0xff0000,
            emissiveIntensity: 0.3,
            metalness: 0.3,
            roughness: 0.4
        });

        // 为矩形的每条边创建一个3D Box
        for (let i = 0; i < 4; i++) {
            const start = normalized[i];
            const end = normalized[(i + 1) % 4];

            // 计算线段的长度和角度
            const dx = end.x - start.x;
            const dy = end.y - start.y;
            const length = Math.sqrt(dx * dx + dy * dy);
            const angle = Math.atan2(dy, dx);

            // 创建3D盒子几何体 (长度 × 厚度 × 深度)
            const geometry = new THREE.BoxGeometry(length, lineThickness, lineDepth);
            const mesh = new THREE.Mesh(geometry, material);

            // 设置位置和旋转
            mesh.position.x = (start.x + end.x) / 2;
            mesh.position.y = (start.y + end.y) / 2;
            mesh.position.z = 0;
            mesh.rotation.z = angle;

            group.add(mesh);
        }

        group.position.z = zPosition; // 与切片同一位置

        return group;
    }

    async loadFrames() {
        const textureLoader = new THREE.TextureLoader();

        for (let i = 0; i < this.frameCount; i++) {
            const framePath = `${this.framesDir}/frame_${String(i).padStart(4, '0')}.png`;

            try {
                const texture = await new Promise((resolve, reject) => {
                    textureLoader.load(
                        framePath,
                        (tex) => {
                            tex.minFilter = THREE.LinearFilter;
                            tex.magFilter = THREE.LinearFilter;
                            resolve(tex);
                        },
                        undefined,
                        reject
                    );
                });

                // 直接使用索引i作为帧索引
                // 因为每个提取的帧对应一个3D切片
                const frameIndex = i;
                const time = i / this.fps;

                this.frameTextures.push({
                    index: frameIndex,
                    texture: texture,
                    time: time
                });

                console.log(`  ✅ 加载帧 ${i}: ${framePath} -> frameIndex=${frameIndex}`);

            } catch (error) {
                console.warn(`⚠️ 无法加载帧 ${framePath}:`, error);
            }
        }

        if (this.frameTextures.length === 0) {
            throw new Error('没有加载到任何帧图片');
        }

        console.log(`✅ 加载了 ${this.frameTextures.length} 帧`);
    }

    createCardStack() {
        const cardWidth = 3;
        const cardHeight = 3 * (9 / 16);
        const spacing = 0.1;
        const totalDepth = this.frameTextures.length * spacing;

        this.frameTextures.forEach((frameData, index) => {
            const geometry = new THREE.PlaneGeometry(cardWidth, cardHeight);

            const material = new THREE.MeshStandardMaterial({
                map: frameData.texture,
                side: THREE.DoubleSide,
                transparent: true,
                opacity: 0.6,
                metalness: 0.3,
                roughness: 0.4
            });

            const mesh = new THREE.Mesh(geometry, material);
            mesh.position.z = -index * spacing + totalDepth / 2;
            mesh.userData = {
                frameIndex: frameData.index,
                isHighlighted: false
            };

            this.scene.add(mesh);
            this.cards.push(mesh);

            // 为每个切片添加红色矩形框
            const redBox = this.createRedBoundingBox(cardWidth, cardHeight, mesh.position.z);
            if (redBox) {
                this.scene.add(redBox);
                this.boundingBoxes.push(redBox);
            }
        });

        console.log(`✅ 创建了 ${this.cards.length} 个3D切片`);
        if (this.boundingBoxes.length > 0) {
            console.log(`✅ 为每个切片添加了红色矩形框`);
        }
    }

    bindVideoEvents() {
        if (!this.videoElement) return;

        const updateHighlight = () => {
            const currentFrame = Math.floor(this.videoElement.currentTime * this.fps);
            this.highlightCard(currentFrame);
        };

        this.videoElement.addEventListener('timeupdate', updateHighlight);
        this.videoElement.addEventListener('seeking', () => updateHighlight());
        this.videoElement.addEventListener('seeked', () => updateHighlight());
    }

    highlightCard(frameIndex, forceUpdate = false) {
        let targetCardIndex = -1;
        for (let i = 0; i < this.cards.length; i++) {
            if (this.cards[i].userData.frameIndex <= frameIndex) {
                targetCardIndex = i;
            } else {
                break;
            }
        }

        if (targetCardIndex === this.currentHighlightIndex && !forceUpdate) {
            return;
        }

        // 恢复之前的高亮卡片
        if (this.currentHighlightIndex >= 0 && this.currentHighlightIndex < this.cards.length) {
            const prevCard = this.cards[this.currentHighlightIndex];
            if (prevCard.userData.frameIndex === this.highlightFrame) {
                prevCard.material.emissive = new THREE.Color(0xff00ff);
                prevCard.material.emissiveIntensity = 1.2;
            } else {
                prevCard.material.emissive = new THREE.Color(0x000000);
                prevCard.material.emissiveIntensity = 0;
            }
            prevCard.userData.isHighlighted = false;
        }

        // 高亮新卡片
        if (targetCardIndex >= 0 && targetCardIndex < this.cards.length) {
            const card = this.cards[targetCardIndex];
            if (card.userData.frameIndex === this.highlightFrame) {
                card.material.emissive = new THREE.Color(0xff00ff);
                card.material.emissiveIntensity = 1.5;
            } else {
                card.material.emissive = new THREE.Color(0x00ffff);
                card.material.emissiveIntensity = 0.8;
            }
            card.userData.isHighlighted = true;
            this.currentHighlightIndex = targetCardIndex;
        }

        // 确保持久高亮帧始终保持高亮
        this.cards.forEach(card => {
            if (card.userData.frameIndex === this.highlightFrame &&
                card.userData.frameIndex !== this.cards[targetCardIndex]?.userData.frameIndex) {
                card.material.emissive = new THREE.Color(0xff00ff);
                card.material.emissiveIntensity = 1.2;
            }
        });
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.composer.render();
    }

    destroy() {
        if (!this.isInitialized) return;

        // 清理卡片
        this.cards.forEach(card => {
            this.scene.remove(card);
            card.geometry.dispose();
            card.material.dispose();
        });

        // 清理红色矩形框
        this.boundingBoxes.forEach(box => {
            this.scene.remove(box);
            box.children.forEach(child => {
                child.geometry.dispose();
                child.material.dispose();
            });
        });

        this.frameTextures.forEach(frameData => {
            frameData.texture.dispose();
        });

        this.renderer.dispose();
        this.composer.dispose();

        this.isInitialized = false;
        console.log('✅ 3D显示已清理');
    }
}

// ES6 模块导出（用于动态导入）
export default Video3DModal;

// 同时也导出为全局变量（兼容旧代码）
if (typeof window !== 'undefined') {
    window.Video3DModal = Video3DModal;
}
